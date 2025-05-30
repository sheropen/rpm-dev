import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict
import math
import json
import os
from transformers import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
import re

from peft import LoraModel, PeftModelForCausalLM
from peft.tuners.lora import LoraLayer
from peft import peft_model

from fastlora.config import FastLoraConfig

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

def _get_layer_idx(name):
    # the name should match the pattern "model.[(model)*].layers.(layer_idx)" (at least 1 "model" in the name)
    # if not match, throw an error
    import re
    match = re.match(r"^(?:model\.)*layers\.(\d+)", name)
    assert match is not None, f"Invalid layer name: {name}"
    return int(match.group(1))

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=torch.float32):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps
    
    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class BatchNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=torch.float32):
        """
        BatchNorm compute the mean and variance for each hidden dimension
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.variance_epsilon = eps
    
    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, hidden_states):
        # hidden_states: B, N, D
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(dim=-2, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(dim=-2, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states + self.bias).to(input_dtype)

class FastLoraLinear(nn.Module, LoraLayer):


    def __init__(
        self,
        base_layer,
        # parent,
        # in_features: int,
        # out_features: int,
        hidden_size: int,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        fastlora_r: int = 0,
        fastlora_max_rank: int = 0,
        fastlora_inter_size: int = None,
        fastlora_alpha: float = 1.0,
        fastlora_dropout: float = 0.0,
        fastlora_arch: str = "as",
        fastlora_norm: str = "rss",
        fastlora_init: str = "random",
        fastlora_merge: str = "mean",
        fastlora_training_attention_mask: Optional[str] = None,
    ):
        nn.Module.__init__(self)

        self.adapter_layer_names = (
            "fastlora_A1", "fastlora_A2", "fastlora_A3", "fastlora_B", 
            "fastlora_hidden_state_norm"
        )
        self.other_param_names = (
            "fastlora_r", "fastlora_max_rank", "fastlora_inter_size", "fastlora_alpha", 
            "fastlora_dropout", "fastlora_arch", "fastlora_norm", "fastlora_init"
        )

        # self.parent = parent
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.hidden_size = hidden_size
        
        self.lora_r = lora_r
        assert lora_r == 0, "FastLoraLinear does not support Lora"

        self.fastlora_r = fastlora_r
        if fastlora_r > 0:
            self.fastlora_max_rank = fastlora_max_rank
            self.fastlora_inter_size = fastlora_inter_size if fastlora_inter_size is not None else self.fastlora_r
            self.fastlora_alpha = fastlora_alpha
            self.fastlora_scaling = fastlora_alpha / fastlora_r
            self.fastlora_arch = fastlora_arch
            self.fastlora_norm = fastlora_norm
            self.fastlora_init = fastlora_init
            self.fastlora_merge = fastlora_merge
            self.fastlora_training_attention_mask = fastlora_training_attention_mask
            self.fastlora_dropout = nn.Dropout(p=fastlora_dropout)
            dtype = self.base_layer.weight.dtype
            if "batchnorm" in fastlora_norm:
                self.fastlora_hidden_state_norm = BatchNorm(hidden_size, dtype=dtype)
            else:
                self.fastlora_hidden_state_norm = RMSNorm(hidden_size, dtype=dtype)
            self.fastlora_A1 = nn.Linear(self.in_features, self.fastlora_inter_size, bias=False, dtype=dtype)
            self.fastlora_A2 = nn.Linear(self.hidden_size, self.fastlora_inter_size, bias=False, dtype=dtype)
            self.fastlora_A3 = nn.Linear(self.hidden_size, self.fastlora_r, bias=False, dtype=dtype)
            self.fastlora_B = nn.Linear(self.fastlora_r, self.out_features, bias=False, dtype=dtype)
            if self.fastlora_norm == "attention":
                self.fastlora_AQ = nn.Linear(self.hidden_size, self.fastlora_max_rank, bias=False, dtype=dtype)
                self.adapter_layer_names = self.adapter_layer_names + ("fastlora_AQ",)
            self.reset_fastlora_parameters()
        
        device = self.base_layer.weight.device
        self.to(device)

    def set_adapter(self, adapter_names: str | list[str]) -> None:
        assert self._active_adapter == "default", "Only one adapter is supported in FastLoraLinear"

        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Deactivate grads on the inactive adapter and activate grads on the active adapter
        for layer_name in self.adapter_layer_names:
            layer = getattr(self, layer_name)
            layer.requires_grad_(True)

        self._active_adapter = adapter_names

    def delete_adapter(self, adapter_names: str | list[str]) -> None:
        raise NotImplementedError("delete_adapter is not supported in FastLoraLinear")

    def _init_prameters(self, kwargs=None):
        self.reset_lora_parameters()
        self.reset_fastlora_parameters(kwargs)

    def reset_fastlora_parameters(self, kwargs=None):
        if self.fastlora_r > 0:
            # if fastlora_hidden_state_norm has weight and bias, then init it
            self.fastlora_hidden_state_norm.reset_parameters()

            nn.init.kaiming_normal_(self.fastlora_A1.weight, mode='fan_in', a=math.sqrt(5))
            nn.init.kaiming_normal_(self.fastlora_A2.weight, mode='fan_in', a=math.sqrt(5))
            nn.init.kaiming_normal_(self.fastlora_A3.weight, mode='fan_in', a=math.sqrt(5))
            nn.init.zeros_(self.fastlora_B.weight)
            if hasattr(self, "fastlora_AQ"):
                nn.init.kaiming_normal_(self.fastlora_AQ.weight, mode='fan_in', a=math.sqrt(5))
            if self.fastlora_init == "copying" and kwargs is not None:
                print("Initializing fastlora_A2 and fastlora_A3 with base model weights")
                assert kwargs is not None, "kwargs is required for copying"
                assert "self_attn.k_proj" in kwargs, "self_attn.k_proj is required for copying"
                assert "self_attn.k_proj" in kwargs, "self_attn.k_proj is required for copying"
                assert self.fastlora_A2.weight.shape == kwargs["self_attn.k_proj"].weight.shape, f"self_attn.k_proj.shape: {kwargs['self_attn.k_proj'].weight.shape}, fastlora_A2.shape: {self.fastlora_A2.weight.shape}"
                assert self.fastlora_A3.weight.shape == kwargs["self_attn.v_proj"].weight.shape, f"self_attn.v_proj.shape: {kwargs['self_attn.v_proj'].weight.shape}, fastlora_A3.shape: {self.fastlora_A3.weight.shape}"
                self.fastlora_A2.weight.data.copy_(kwargs["self_attn.k_proj"].weight.data)
                self.fastlora_A3.weight.data.copy_(kwargs["self_attn.v_proj"].weight.data)
    
    def _rms_norm(self, hidden_states, variance_epsilon=1e-6):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance_sum = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance_sum + variance_epsilon)
        return hidden_states.to(input_dtype)

    def _frobenius_norm(self, M, variance_epsilon=1e-6):
        # shape: ... x DI x DO
        input_dtpye = M.dtype
        DI, DO = M.shape[-2], M.shape[-1]
        # print(M)
        M = M.to(torch.float32)
        variance_sum = M.pow(2).sum(dim=(-1, -2), keepdim=True)
        M = M * torch.rsqrt(variance_sum + variance_epsilon)
        # print(DI, DO, variance_sum.mean())
        return M.to(input_dtpye)
    
    def _spectral_norm(self, M, epsilon=1e-6):
        # shape: ... x DI x DO
        input_dtpye = M.dtype
        DI, DO = M.shape[-2], M.shape[-1]
        M = M.to(torch.float32)
        S = torch.linalg.svdvals(M)    # ... x min(DI, DO)
        # print(S.shape, S.mean(), S.max(), S.min())
        max_s = S[..., 0]   # ... x 1
        # print("spectral", max_s)
        # print("spectral 2 norm", torch.sqrt(S.pow(2).sum(dim=-1)))
        # print("frobenius", torch.sqrt(M.pow(2).sum(dim=(-1, -2))))
        M = M / (max_s.unsqueeze(-1).unsqueeze(-1) + epsilon)
        return M.to(input_dtpye)

    def _svd_norm(self, M, reset_threshold=1e-7, reset=True):
        # shape: B ... x DI x DO
        input_dtpye = M.dtype
        M_f32 = M.to(torch.float32)
        # # case 1: if it is test time or the matrix is small, we do the full svd
        # if min(M_f32.shape[-2], M_f32.shape[-1]) <= self.fastlora_max_rank or (not self.training):
        #     U, S, Vh = torch.linalg.svd(M_f32, full_matrices=False)
        #     U = U[..., :self.fastlora_max_rank]
        #     S = S[..., :self.fastlora_max_rank]
        #     V = Vh.transpose(-2, -1)
        #     V = V[..., :self.fastlora_max_rank]
        #     # print(S.mean(), S.max(), S.min())
        #     # print(S[:, :128].mean(), S[:, :128].sort(descending=True).values[0, :4], S[:, :128].sort(descending=True).values[0, -4:])
        # # otherwise, we use the randomized svd
        # else:
        #     U, S, V = torch.svd_lowrank(M_f32, q=self.fastlora_max_rank, niter=1)
        #     # print(S.mean(), S.max(), S.min())
        #     # print(S[:, :128].mean(), S[:, :128].sort(descending=True).values[0, :4], S[:, :128].sort(descending=True).values[0, -4:])
        try:
            U, S, V = torch.svd_lowrank(M_f32, q=self.fastlora_max_rank, niter=1)
        except Exception as e:
            print(f"[WARNING] SVD failed, resetting the adaptor to zero, error: {e}")
            print("M_f32.shape", M_f32.shape)
            print("M_f32", M_f32)
            raise e
            # return torch.zeros_like(M)

        Vh = V.transpose(-2, -1)
        M_norm = U @ Vh   # B ... x DI x DO

        # print(M.shape, M_norm.shape, U.shape, S.shape, Vh.shape)
        # print(M_norm)

        # during training if S contains similar singular values, the gradient is not stable. if that happens, we just drop the adaptor and return zero.
        if self.training and reset:
            # Compute the minimum separation between consecutive singular values
            eps = 1e-6
            min_separation = torch.minimum((S[..., :-1] - S[..., 1:]) / (S[..., 0].unsqueeze(-1) + eps), (S[..., :-1] - S[..., 1:])).min(dim=-1).values

            # Create a boolean mask where the minimum separation is less than the threshold
            reset_mask = min_separation < reset_threshold

            # Expand the mask dimensions for broadcasting
            reset_mask = reset_mask[..., None, None]

            # Reset M_norm to zero where the mask is True
            M_norm = M_norm.masked_fill(reset_mask, 0.0)

            if reset_mask.any():
                print(f"[WARNING] Singular values are too close, resetting the adaptor to zero, min_separation: {min_separation}")

        return M_norm.to(input_dtpye)

    def _lstsq(self, A, B, lamb=1e-2):
        """
        Differentiable least squares
        :param A: ... x m x n
        :param B: ... x m x p
        """
        assert A.dtype == B.dtype, f"A.dtype: {A.dtype}, B.dtype: {B.dtype}"
        assert A.dim() == 3 and B.dim() == 3, f"A.dim(): {A.dim()}, B.dim(): {B.dim()}"
        
        input_dtype = A.dtype
        cols = A.shape[-1]
        
        # Cast to float32 for numerical stability
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        
        # Compute A^T * A + lambda * I
        A_dash = A.transpose(-2, -1) @ A + lamb * torch.eye(cols, device=A.device)
        
        # Compute A^T * B
        B_dash = A.transpose(-2, -1) @ B
        
        # Solve the linear system
        output = torch.linalg.solve(A_dash, B_dash)
        
        # Return the result in the original data type
        return output.to(input_dtype)

    def _merge(self, x, is_causal=False, merge_type='mean'):
        # x: B, S1, S2, ...
        assert x.dim() == 5, f"x.dim(): {x.dim()}"
        B, S1, S2 = x.shape[:3]
        segment_merge_mask = torch.ones(S1, S2, device=x.device).bool()    # S1, S2
        if is_causal:
            segment_merge_mask = torch.tril(segment_merge_mask, diagonal=-1)
            if self.fastlora_training_attention_mask and self.fastlora_training_attention_mask.startswith("abcdabcd") and self.training:
                # NOTE: this is a special case for the training attention mask "abcdabcd"
                assert S1 == 16 and S2 == 16, f"S1: {S1}, S2: {S2}"
                # S[1][0] = 1, 
                # S[2][0, 1] = 1
                # ...
                # S[7][0, 1, 2, 3, 4, 5, 6] = 1
                # S[8][0] = 1
                # S[9][0, 1] = 1
                # ...
                # S[15][0, 1, 2, 3, 4, 5, 6, 7] = 1
                segment_merge_mask[8:15] = segment_merge_mask[1:8]
                segment_merge_mask[15, 8:] = 0

        segment_merge_mask = segment_merge_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(x.shape)   # B, S1, S2, ...
        # print(segment_merge_mask)
        
        if merge_type == "sum":
            x = x * segment_merge_mask  # B, S1, S2, ...
            x = x.sum(dim=2)    # B, S1, ...
        elif merge_type == "mean":
            x = x * segment_merge_mask  # B, S1, S2, ...
            x = x.sum(dim=2) / torch.clamp(segment_merge_mask.sum(dim=2), min=1)   # B, S1, ...
        else:
            raise ValueError(f"Unknown fastlora_merge: {merge_type}")
        return x

    def _normalize(self, ss):
        if self.fastlora_norm == "frobenius":
            ss = self._frobenius_norm(ss)
        elif self.fastlora_norm == "spectral":
            ss = self._spectral_norm(ss)
        elif self.fastlora_norm == "svd":
            ss = self._svd_norm(ss)
        else:
            raise ValueError(f"Unknown fastlora_norm: {self.fastlora_norm}")
        return ss


    def _kv_mapping_matrix(self, hidden_states_norm, attention_mask, outer_product=None, merge_target_size=None, merge_is_caual=False):
        assert merge_target_size is not None, "merge_target_size is required for post-norm"
        input_dtype = hidden_states_norm.dtype

        if self.fastlora_norm == "attention":
            raise NotImplementedError("Attention norm is not implemented yet")
        
            # self-attention between hidden_states_norm and fastlora_AQ
            c_centroids = self.fastlora_AQ(hidden_states_norm)   # B, S2, L2, RM
            c_centroids = torch.where(attention_mask.unsqueeze(-1).bool(), c_centroids, float("-inf"))   # B, S2, L2, RM
            c_centroids = F.softmax(c_centroids, dim=-2)   # B, S2, L2, RM
            # compute key and value states
            c_key_states = c_centroids.transpose(-2, -1) @ hidden_states_norm    # B, S2, RM, D
            c_key_states = self.fastlora_A2(c_key_states)   # B, S2, RM, R1
            c_value_states = c_centroids.transpose(-2, -1) @ hidden_states_norm    # B, S2, RM, D
            c_value_states = self.fastlora_A3(c_value_states)   # B, S2, RM, R2
            ss = c_key_states.transpose(-2, -1) @ c_value_states    # B, S2, R1, R2
            
            ss = self._frobenius_norm(ss)

        else:
            c_key_states = self.fastlora_A2(hidden_states_norm)   # B, S2, L2, R1
            c_value_states = self.fastlora_A3(hidden_states_norm)    # B, S2, L2, R2
            c_key_states = c_key_states * attention_mask.unsqueeze(-1)   # B, S2, L2, R1
            c_value_states = c_value_states * attention_mask.unsqueeze(-1)  # B, S2, L2, R2
            ss = c_key_states.transpose(-2, -1) @ c_value_states    # B, S2, R1, R2
            # tokens_count = attention_mask.sum(dim=-1, keepdim=True).unsqueeze(-1)    # B, S2, 1, 1
            # ss = (ss.to(torch.float32) / torch.sqrt(tokens_count))  # B, S2, R1, R2

            if self.fastlora_merge.startswith("pre-norm-"):
                ss = ss.unsqueeze(1).expand(-1, merge_target_size, -1, -1, -1)    # B, S1, S2, R1, R2
                ss = self._merge(
                    ss,  # B, S1, S2, R1, R2
                    is_causal=merge_is_caual,
                    merge_type=self.fastlora_merge.removeprefix("pre-norm-"),
                )   # B, S1, R1, R2
                
                if outer_product is not None:
                    ss = ss + outer_product.unsqueeze(1).expand(-1, merge_target_size, -1, -1)

                if merge_is_caual == False:
                    # mode is states, we only need to normalize the state once
                    ss_pre = ss[:, :1, :, :]    # B, 1, R1, R2
                    ss_post = self._normalize(ss_pre)   # B, 1, R1, R2
                    ss = ss_post.expand(-1, merge_target_size, -1, -1)  # B, S1, R1, R2
                else:
                    if self.fastlora_training_attention_mask and self.fastlora_training_attention_mask.startswith("abcdabcd") and self.training:
                        # NOTE: here is a hack for the training attention mask "abcdabcd"
                        assert merge_target_size == 16 and hidden_states_norm.shape[1] == 16
                        ss_pre = ss[:, 8:16, :, :]    # B, 8, R1, R2
                        ss_post = self._normalize(ss_pre)   # B, 8, R1, R2
                        ss = torch.cat([torch.zeros_like(ss[:, :1]), ss_post[:, :7], ss_post], dim=1)   # B, 16, R1, R2
                    else:
                        # mode is default, the first segment does not have fora
                        ss_pre = ss[:, 1:, :, :]    # B, S1-1, R1, R2
                        ss_post = self._normalize(ss_pre)   # B, S1-1, R1, R2
                        ss = torch.cat([torch.zeros_like(ss[:, :1]), ss_post], dim=1)    # B, S1, R1, R2
                    
            else:
                raise NotImplementedError("post-norm is not implemented yet")
                ss = self._normalize(ss)    # B, S2, R1, R2
                raise ValueError(f"Unknown fastlora_norm: {self.fastlora_norm}")
            
                if not self.fastlora_merge.startswith("pre-norm-"):
                    norm_starter = self.fastlora_merge[:9]
                    ss = ss.unsqueeze(1).expand(-1, merge_target_size, -1, -1, -1)    # B, S1, S2, R1, R2
                    ss = self._merge(
                        ss,  # B, S1, S2, R1, R2
                        is_causal=merge_is_caual,
                        merge_type=self.fastlora_merge.removeprefix(norm_starter),
                    )   # B, S1, R1, R2

                if self.fastlora_merge.startswith("mix-norm-"):
                    if self.fastlora_norm == "frobenius":
                        ss = self._frobenius_norm(ss)
                    elif self.fastlora_norm == "spectral":
                        ss = self._spectral_norm(ss)
                    elif self.fastlora_norm == "svd":
                        ss = self._svd_norm(ss, reset=False)
                    else:
                        raise ValueError(f"Unknown fastlora_norm: {self.fastlora_norm}")

        return ss.to(input_dtype)

    def _x_transform(self, x, hidden_states, attention_mask, mode="default"):
        # input vectors x (x.shape: B, S1, L1, DI)
        # hidden states of context hidden_states (C.shape: B, S2, L2, D)
        # mask of hidden states attention_mask (M.shape: B, S2, L2)
        x_input = x
        B, S1, L1, _ = x.shape
        B, S2, L2, _ = hidden_states.shape
        input_dtype = x.dtype

        if mode == "default":
            assert S1 == S2, "S1 should be equal to S2 in default mode"
            assert L1 == L2, "L1 should be equal to L2 in default mode"
            # # the last segment in hidden_states does not contribute to the output
            # hidden_states = hidden_states[:, :-1]
            # attention_mask = attention_mask[:, :-1]
            # B, S2, L2, _ = hidden_states.shape
            # # the first segment in x does not change
            # x = x[:, 1:]
            # B, S1, L1, _ = x.shape

        # Step 1: normalize the hidden states
        hidden_states_norm = self.fastlora_hidden_state_norm(hidden_states)

        # Step 2: compute a segment-wise transformation of x (output: B, S1, S2, L1, Do)
        if self.fastlora_norm == "softmax":
            x_query_states = self.fastlora_A1(x)     # B, S1, L1, R1
            c_key_states = self.fastlora_A2(hidden_states_norm)   # B, S2, L2, R1
            c_value_states = self.fastlora_A3(hidden_states_norm)    # B, S2, L2, R2
            # print("x_query_states", x_query_states[:, 1, :8, :8])
            # print("c_key_states", c_key_states[:, 0, :8, :8])
            # print("c_value_states", c_value_states[:, 0, :8, :8])
            x_query_states = x_query_states.unsqueeze(2).expand(-1, -1, S2, -1, -1)   # B, S1, S2, L1, R1
            c_key_states = c_key_states.unsqueeze(1).expand(-1, S1, -1, -1, -1)    # B, S1, S2, L2, R1
            c_value_states = c_value_states.unsqueeze(1).expand(-1, S1, -1, -1, -1)    # B, S1, S2, L2, R2
            # convert attention_mask (B, S2, L2) to attn_mask with shape: # B, S1, S2, L1, L2
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(3).expand(-1, S1, -1, L1, -1).bool()   # B, S1, S2, L1, L2
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                x_query_states,
                c_key_states,
                c_value_states,
                attn_mask=attn_mask,
                dropout_p=0.0, is_causal=False,
            )   # B, S1, S2, L1, R2
            # attn_output = attn_output.reshape(B, S1, S2, L1, -1)    # B, S1, S2, L1, R2
            # print("attn_output", attn_output[:, 1, 0, :8, :8])
            x = self.fastlora_B(attn_output)    # B, S1, S2, L1, Do

            if not self.fastlora_merge.startswith("pre-norm"):
                x = self._merge(
                    x,  # B, S1, S2, L1, Do
                    is_causal=True if mode == 'default' else False,
                    merge_type=self.fastlora_merge,
                )   # B, S1, L1, Do
            else:
                raise ValueError("pre-norm is not supported in softmax mode")
        else:
            x_query_states = self.fastlora_A1(x)     # B, S1, L1, R1

            ss = self._kv_mapping_matrix(
                hidden_states_norm, attention_mask,
                merge_target_size=S1,
                merge_is_caual=True if mode == 'default' else False
            )    # B, S1, R1, R2

            x = x_query_states @ ss    # B, S1, L1, R2
            x = self.fastlora_B(x)    # B, S1, L1, Do
        
        # if mode == "default":
        #     x = torch.concat([x_input[:, :1], x], dim=1)    # B, (1 + S1), L1, Do

        assert x.dtype == input_dtype, f"x.dtype: {x.dtype}, input_dtype: {input_dtype}"
        return x

    def forward(self, x: torch.Tensor):
        # print(f">>> calling FastLoraLinear.forward (pid={os.getpid()})", flush=True)

        # print(past_hidden_states.shape if past_hidden_states is not None else None)
        x_input = x
        B, S = self.args["batch_size"], self.args["num_segments"]
        x = x.reshape((B, S) + x.shape[1:])

        # print("x_0", x[0, 0, :8, :8])
        # print("x_1", x[0, 1, :8, :8])
        # print("hidden_states_0", self.args["hidden_states"][0, 0, :8, :8])
        # print("hidden_states_1", self.args["hidden_states"][0, 1, :8, :8])

        result = self.base_layer(x)   # B x S x L x Do

        if self.fastlora_r > 0:
            mode = self.args["mode"]
            if mode == "weights":
                lora_a = self.args["lora_a"]    # B x R x Di
                lora_b = self.args["lora_b"]    # B x Do x R
                # x.shape: B x S x L x Di
                x = x @ lora_a.unsqueeze(1).transpose(-2, -1)    # B x S x L x R
                x = x @ lora_b.unsqueeze(1).transpose(-2, -1)    # B x S x L x Do
                result = result + self.fastlora_scaling * x   # B x S x L x Do
            elif mode == "default" or mode == "states":
                hidden_states = self.args["hidden_states"]    # B x S x L x D
                attention_mask = self.args["attention_mask"]    # B x S x L
                x = self._x_transform(x, hidden_states, attention_mask, mode=mode)  # B x S x L x Do
                result = result + self.fastlora_scaling * x  # B x S x L x Do
                # print("delta_x_0", x[0, 0, :8, :8])
                # print("delta_x_1", x[0, 1, :8, :8])   
            else:
                raise ValueError(f"Unknown mode: {mode}")

        result = result.reshape((B * S,) + result.shape[2:])

        assert result.dtype == x_input.dtype, f"result.dtype: {result.dtype}, x_input.dtype: {x_input.dtype}"
        # print("<<< calling FastLoraLinear.forward", flush=True)
        return result

    @torch.inference_mode()
    def to_lora_weights(self, hidden_states=None, attention_mask=None, outer_product=None, return_outer_product=False):
        """
            Convert the FastLoraLinear layer to Lora weights
            hidden_states: B, S, L, D
            attention_mask: B, S, L
            output should be a dictionary with keys "lora_a" and "lora_b"
                - lora_a: B, R, Di
                - lora_b: B, Do, R
        """
        # Step 1: normalize the hidden states
        hidden_states_norm = self.fastlora_hidden_state_norm(hidden_states)   # B, S, L, D

        ss = self._kv_mapping_matrix(
            hidden_states_norm, attention_mask,
            outer_product=outer_product,
            merge_target_size=1,
            merge_is_caual=False
        )    # B, 1, R1, R2

        A1 = self.fastlora_A1.weight.data   #  R1, Di
        B = self.fastlora_B.weight.data   #  Do, R2

        A = ss.squeeze(1).transpose(-2, -1) @ A1.unsqueeze(0)    # B, S, R2, Di

        B = B.unsqueeze(0).expand(A.shape[0], -1, -1)    # B, Do, R2

        if return_outer_product:
            c_key_states = self.fastlora_A2(hidden_states_norm)   # B, S2, L2, R1
            c_value_states = self.fastlora_A3(hidden_states_norm)    # B, S2, L2, R2
            c_key_states = c_key_states * attention_mask.unsqueeze(-1)   # B, S2, L2, R1
            c_value_states = c_value_states * attention_mask.unsqueeze(-1)  # B, S2, L2, R2
            ss = c_key_states.transpose(-2, -1) @ c_value_states    # B, S2, R1, R2
            assert self.fastlora_merge == "pre-norm-sum", f"fastlora_norm: {self.fastlora_norm}"
            ss = ss.sum(dim=1)    # B, R1, R2
            if outer_product is not None:
                assert outer_product.shape == ss.shape, f"outer_product.shape: {outer_product.shape}, ss.shape: {ss.shape}"
                ss = ss + outer_product
            return {"lora_a": A, "lora_b": B, "outer_product": ss}

        return {"lora_a": A, "lora_b": B}        

class FastLoraDecoderLayer(nn.Module):
    def __init__(self, layer, target_modules=None):
        super().__init__()
        self.base_layer = layer
        self.update_target_modules(target_modules)
        # print(target_modules)

    def update_target_modules(self, target_modules):
        self._target_modules = target_modules if target_modules is not None else []

    def forward(self, hidden_states, attention_mask, *args, **kwargs):

        for target_module in self._target_modules:
            if target_module.args["mode"] == "default":
                B, S = target_module.args["batch_size"], target_module.args["num_segments"]
                # print(B, S, hidden_states.shape)
                target_module.args["hidden_states"] = hidden_states.reshape((B, S) + hidden_states.shape[1:])
        
        # print(f">>> calling FastLoraDecoderLayer.forward (pid={os.getpid()})", flush=True)
        outputs = self.base_layer(hidden_states, attention_mask, *args, **kwargs)
        # print("<<< calling FastLoraDecoderLayer.forward", flush=True)

        # for target_module in self._target_modules:
        #     if target_module.args["mode"] == "default":
        #         target_module.args["hidden_states"] = None
        

        return outputs

class FastLoraModel(LoraModel):
    
    prefix: str = "fastlora_"

    def __init__(self, model, config, adapter_name='default'):
        fastlora_config = config[adapter_name]
        # check config
        if fastlora_config.target_modules is None and fastlora_config.fastlora_param is not None:
            fastlora_config.target_modules = fastlora_config.fastlora_param
        assert fastlora_config.lora_r == 0, "FastLoraModel does not support Lora"

        layer_names = [name for name, _ in model.named_modules() if re.match(r"^(?:model\.)*layers\.\d+$", name)]
        for layer_name in layer_names:
            parent, layer_module, target_name = _get_submodules(model, layer_name)
            new_layer_module = FastLoraDecoderLayer(layer_module)
            self._replace_module(parent, target_name, new_layer_module, layer_module)

        super().__init__(model, config, adapter_name)

        # self.targeted_module_names
        for layer_name in layer_names:
            layer_module = model.get_submodule(layer_name)
            targted_modules = [self.model.get_submodule(name) for name in self.targeted_module_names if name.startswith(layer_name + ".")]
            layer_module.update_target_modules(targted_modules)
        
        # print([name for name, _ in model.named_modules()])
        # print(layer_names)
        # print(self.targeted_module_names)

    # def inject_adapter(self, model):
    #     key_list = [key for key, _ in model.named_modules()]
    #     print(key_list)
    #     for key in key_list:
    #         if not _check_target_module_exists(self.fastlora_config, key):
    #             continue

    #         self.targeted_module_names.append(key)
    #         parent, target, target_name = _get_submodules(model, key)
    #         self._create_and_replace(self.fastlora_config, target, target_name, parent, key)
    #     self._mark_only_adapters_as_trainable(model)
    
    def _create_and_replace(self, fastlora_config, adapter_name, target, target_name, parent, current_key):
        kwargs = {
            # "parent": parent,
            # "in_features": target.in_features,
            # "out_features": target.out_features,
            "hidden_size": self.model.config.hidden_size,
            "lora_r": fastlora_config.lora_r,
            "lora_alpha": fastlora_config.lora_alpha,
            "lora_dropout": fastlora_config.lora_dropout,
            "fastlora_r": fastlora_config.fastlora_r,
            "fastlora_max_rank": fastlora_config.fastlora_max_rank,
            "fastlora_inter_size": fastlora_config.fastlora_inter_size,
            "fastlora_alpha": fastlora_config.fastlora_alpha,
            "fastlora_dropout": fastlora_config.fastlora_dropout,
            "fastlora_arch": fastlora_config.fastlora_arch,
            "fastlora_norm": fastlora_config.fastlora_norm,
            "fastlora_init": fastlora_config.fastlora_init,
            "fastlora_merge": fastlora_config.fastlora_merge,
            "fastlora_training_attention_mask": fastlora_config.fastlora_training_attention_mask,
        }
        new_module = self._create_new_module(fastlora_config, adapter_name, target, **kwargs)
        if adapter_name not in self.active_adapters:
            # adding an additional adapter: it is not automatically trainable
            new_module.requires_grad_(False)
        self._replace_module(parent, target_name, new_module, target)

    def _create_new_module(self, fastlora_config, adapter_name, target, **kwargs):
        return FastLoraLinear(target, **kwargs)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)

    # def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
    #     for name, module in model.named_modules():
    #         if self.prefix in name:
    #             print(name)
    #             module.requires_grad_(True)
    #         else:
    #             module.requires_grad_(False)
class FastLoraModelForCausalLM(PeftModelForCausalLM):


    def _before_forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        fastlora_hidden_states_and_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        fastlora_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        assert input_ids is not None
            
        if input_ids.dim() == 3:
            B, S, L = input_ids.shape
        elif input_ids.dim() == 2:
            B, L = input_ids.shape
            S = 1
        else:
            raise ValueError(f"Invalid input_ids shape: {input_ids.shape}")

        if fastlora_hidden_states_and_mask is not None:
            mode = "states"
            assert input_ids.dim() == 2, "input_ids should be 2D tensor in states mode"
            assert fastlora_weights is None, "fastlora_weights should be None in states mode"
            fastlora_hidden_states, fastlora_attension_mask = fastlora_hidden_states_and_mask
            assert fastlora_hidden_states[0].dim() == 4, "fastlora_hidden_states should be 4D tensor (batch_size x num_segments x seq_len x hidden_size)"
            assert fastlora_attension_mask.dim() == 3, "fastlora_mask should be 3D tensor (batch_size x num_segments x seq_len)"
            assert fastlora_hidden_states[0].shape[:-1] == fastlora_attension_mask.shape, "fastlora_hidden_states and fastlora_mask should have the same shape"
        elif fastlora_weights is not None:
            mode = "weights"
            assert input_ids.dim() == 2, "input_ids should be 2D tensor in weight mode"
            assert fastlora_hidden_states_and_mask is None, "fastlora_hidden_states should be None in weight mode"
        else:
            mode = "default"
            # in compression and training mode, the shape of input_ids is #batch_size x #num_segments x #seq_len
            # past_key_values, fastlora_hidden_states, and fastlora_weights should be zero
            assert input_ids.dim() == 2 or input_ids.dim() == 3, "input_ids should be 2D or 3D tensor in default mode"
            assert (past_key_values is None) and (fastlora_hidden_states_and_mask is None) and (fastlora_weights is None), "past_key_values, fastlora_hidden_states, and fastlora_weights should be None"

            fastlora_attension_mask = attention_mask
            input_ids = input_ids.reshape(B * S, L)
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(B * S, L)
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds.reshape(B * S, L, -1)
            if labels is not None:
                labels = labels.reshape(B * S, L)

        # iterate over all targeted modules. for each, we pass an argument about the fastlora_hidden_states
        for target_module_name in self.base_model.targeted_module_names:
            args_dict = {}
            if mode == "states":
                args_dict["mode"] = "states"
                layer_idx = _get_layer_idx(target_module_name)
                args_dict["hidden_states"] = fastlora_hidden_states[layer_idx]
                args_dict["attention_mask"] = fastlora_attension_mask
            elif mode == "weights":
                args_dict["mode"] = "weights"
                args_dict["lora_a"] = fastlora_weights[target_module_name]["lora_a"]
                args_dict["lora_b"] = fastlora_weights[target_module_name]["lora_b"]
            elif mode =="default":
                args_dict["mode"] = "default"
                args_dict["hidden_states"] = None
                args_dict["attention_mask"] = fastlora_attension_mask
            else:
                raise ValueError(f"Unknown mode: {mode}")
            args_dict["batch_size"] = B
            args_dict["num_segments"] = S
            module = self.base_model.model.get_submodule(target_module_name)
            module.args = args_dict
            # print(f"@@@ set {target_module_name}")
        
        if self.peft_config['default'].fastlora_training_attention_mask and self.peft_config['default'].fastlora_training_attention_mask.startswith("abcdabcd") and self.training:
            if labels is not None:
                assert labels.shape[0] == 16, f"labels.shape: {labels.shape}"
                if self.peft_config['default'].fastlora_training_attention_mask == "abcdabcd-reconstruction":
                    labels[0:8] = -100
                elif self.peft_config['default'].fastlora_training_attention_mask == "abcdabcd-completion":
                    labels[8:16] = -100
            

        return {
            "mode": mode,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "labels": labels,
        }

    def _after_forward(
        self,
        outputs=None,
        input_ids_shape=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # for target_module_name in self.base_model.targeted_module_names:
        #     module = self.base_model.model.get_submodule(target_module_name)
        #     module.args = None
        #     print(f"@@@@ unset {target_module_name}")
        
        if outputs is not None:
            assert input_ids_shape is not None, "input_ids_shape should not be None"
            if "logits" in outputs:
                logits = outputs["logits"]
                outputs["logits"] = logits.reshape(input_ids_shape[:-1] + logits.shape[1:])
            if "past_key_values" in outputs:
                new_past_key_values = tuple(
                    (
                        past_key_value[0].reshape(input_ids_shape[:-1] + past_key_value[0].shape[1:]),
                        past_key_value[1].reshape(input_ids_shape[:-1] + past_key_value[1].shape[1:])
                    )
                    for past_key_value in outputs["past_key_values"]
                )
                outputs["past_key_values"] = new_past_key_values
            if "attentions" in outputs:
                new_attentions = tuple(
                    attention.reshape(input_ids_shape[:-1] + attention.shape[1:])
                    for attention in outputs["attentions"]
                )
                outputs["attentions"] = new_attentions
            if "hidden_states" in outputs:
                new_hidden_states = tuple(
                    hidden_state.reshape(input_ids_shape[:-1] + hidden_state.shape[1:])
                    for hidden_state in outputs["hidden_states"]
                )
                outputs["hidden_states"] = new_hidden_states
        return outputs
                

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        fastlora_hidden_states_and_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        fastlora_weights: Optional[Dict[str, torch.Tensor]] = None,

        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        # print(f">>> calling FastLoraModel.forward (pid={os.getpid()})", flush=True)
        
        input_kwargs = self._before_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            past_key_values=past_key_values,
            fastlora_hidden_states_and_mask=fastlora_hidden_states_and_mask,
            fastlora_weights=fastlora_weights,
        )

        # print(input_kwargs["input_ids"])
        
        outputs = super().forward(
            input_ids=input_kwargs["input_ids"],
            attention_mask=input_kwargs["attention_mask"],
            inputs_embeds=input_kwargs["inputs_embeds"],
            labels=input_kwargs["labels"],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            task_ids=task_ids,
            **kwargs
        )

        # print(outputs.logits.shape)
        # print("logits_0", outputs.logits[0, :8, :8])
        # print("logits_1", outputs.logits[1, :8, :8])
        # torch.gather(outputs.logits[1, :-1][labels[0, 1, 1:] != -100].softmax(dim=-1), dim=1, index=labels[0, 1, 1:][labels[0, 1, 1:] != -100].unsqueeze(-1))

        outputs = self._after_forward(
            outputs=outputs,
            input_ids_shape=input_ids.shape,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # print("<<< done FastLoraModel.forward", flush=True)

        return outputs

    @torch.inference_mode()
    def to_lora_weights(self, hidden_states, attention_mask, outer_product=None, return_outer_product=False):
        """
            hidden_states: (B, S, L, D) x layers
            attention_mask: B, S, L
        """
        assert hidden_states[0].dim() == 4, "hidden_states should be 4D tensor (batch_size x num_segments x seq_len x hidden_size)"
        assert attention_mask.dim() == 3, "attention_mask should be 3D tensor (batch_size x num_segments x seq_len)"
        outer_product = {} if outer_product is None else outer_product
        
        lora_weights = {}
        for target_module_name in self.base_model.targeted_module_names:
            module = self.base_model.model.get_submodule(target_module_name)
            layer_index = _get_layer_idx(target_module_name)
            lora_weights[target_module_name] = module.to_lora_weights(
                hidden_states[layer_index], attention_mask, outer_product=outer_product.get(target_module_name, None), return_outer_product=return_outer_product
            )
        return lora_weights

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None, 
        fastlora_hidden_states_and_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        fastlora_weights: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        
        input_kwargs = self._before_forward(
            input_ids=inputs,
            attention_mask=None,
            inputs_embeds=None,
            labels=None,
            past_key_values=None,
            fastlora_hidden_states_and_mask=fastlora_hidden_states_and_mask,
            fastlora_weights=fastlora_weights,
        )

        outputs = super().generate(
            inputs=input_kwargs["input_ids"],
            **kwargs
        )

        # self._after_forward(
        # )

        return outputs

def get_peft_model_state_dict(model, state_dict=None, adapter_name="default", unwrap_compiled=False, save_embedding_layers="auto"):
    print(f"[WARNING] we apply a monkey patch to the function `get_peft_model_state_dict` to support FastLora model")
    config = model.peft_config[adapter_name]
    assert isinstance(config, FastLoraConfig), f"config should be FastLoraConfig, but got {type(config)}"
    if unwrap_compiled:
        model = getattr(model, "_orig_mod", model)
    if state_dict is None:
        state_dict = model.state_dict()
    to_return = {k: state_dict[k] for k in state_dict if "fastlora_" in k}
    return to_return

def set_peft_model_state_dict(
    model, peft_model_state_dict, adapter_name="default", ignore_mismatched_sizes: bool = False
):
    from peft.utils.save_and_load import _find_mismatched_keys
    import warnings

    print(f"[WARNING] we apply a monkey patch to the function `get_peft_model_state_dict` to support FastLora model")
    config = model.peft_config[adapter_name]
    assert isinstance(config, FastLoraConfig), f"config should be FastLoraConfig, but got {type(config)}"
    state_dict = {}
    state_dict = peft_model_state_dict

    peft_model_state_dict = {}
    parameter_prefix = "fastlora_"
    for k, v in state_dict.items():
        if parameter_prefix in k:
            # suffix = k.split(parameter_prefix)[1]
            # if "." in suffix:
            #     suffix_to_replace = ".".join(suffix.split(".")[1:])
            #     k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
            # else:
            #     k = f"{k}.{adapter_name}"
            peft_model_state_dict[k] = v
        else:
            peft_model_state_dict[k] = v

    missing_keys = set(model.state_dict().keys()) - set(peft_model_state_dict.keys())
    missing_keys = [key for key in missing_keys if "fastlora_" in key]
    print(f"missing_keys: {missing_keys}")
    print(f"peft_model_state_dict.keys(): {peft_model_state_dict.keys()}")

    peft_model_state_dict, mismatched_keys = _find_mismatched_keys(
        model, peft_model_state_dict, ignore_mismatched_sizes=ignore_mismatched_sizes
    )
    load_result = model.load_state_dict(peft_model_state_dict, strict=False)

    if mismatched_keys:
        # see https://github.com/huggingface/transformers/blob/09f9f566de83eef1f13ee83b5a1bbeebde5c80c1/src/transformers/modeling_utils.py#L4039
        mismatched_warning = "\n".join(
            [
                f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                for key, shape1, shape2 in mismatched_keys
            ]
        )
        msg = (
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint "
            f"and are being ignored because you passed `ignore_mismatched_sizes=True`: {mismatched_warning}."
        )
        warnings.warn(msg)
    return load_result

def load_pretrained_model(model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", **kwargs):
    from peft.auto import AutoPeftModelForCausalLM
    from transformers import AutoModelForCausalLM
    # AutoPeftModelForCausalLM._target_peft_class = FastLoraModelForCausalLM
    try:
        # model = AutoPeftModelForCausalLM.from_pretrained(
        #     model_name_or_path, torch_dtype=torch_dtype, attn_implementation=attn_implementation,
        #     **kwargs
        # )
        from peft.config import PeftConfig
        peft_config = PeftConfig.from_pretrained(model_name_or_path, **kwargs)
        base_model_path = peft_config.base_model_name_or_path
        assert base_model_path is not None, "base_model_name_or_path should not be None"
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        peft_config.task_type = "FAST_LORA_CAUSAL_LM"
        if kwargs.get("fastlora_params", None) is not None:
            for key, value in kwargs["fastlora_params"].items():
                setattr(peft_config, key, value)
        print(f'peft_config', peft_config)
        model = FastLoraModelForCausalLM.from_pretrained(
            base_model,
            model_name_or_path,
            adapter_name='default',
            is_trainable=False,
            config=peft_config,
            **kwargs,
        )
    except ValueError:
        model = from_pretrained_v1(
            model_name_or_path, torch_dtype=torch_dtype, attn_implementation=attn_implementation,
            **kwargs
        )
    return model

def from_pretrained_v1(model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None, **kwargs):
    from fastlora.mistral.modeling_mistral import MistralForCausalLM as FastLoraMistralForCausalLM
    from transformers import MistralForCausalLM, AutoConfig
    from peft.utils.save_and_load import _find_mismatched_keys
    import warnings

    ckpt_model = FastLoraMistralForCausalLM.from_pretrained(model_name_or_path)
    ckpt_state_dict = ckpt_model.state_dict()
    ckpt_state_dict = {k: v for k, v in ckpt_state_dict.items() if "fastlora_" in k}
    ckpt_state_dict = {k: v for k, v in ckpt_state_dict.items() if "base_layer" not in k}
    ckpt_state_dict = {k: v for k, v in ckpt_state_dict.items() if "fastlora_embed_tokens" not in k}
    ckpt_state_dict = {k.replace("hidden_state_norm_fn", "fastlora_hidden_state_norm"): v for k, v in ckpt_state_dict.items()}
    ckpt_state_dict = {k.replace(".fastlora_o_proj", ".o_proj"): v for k, v in ckpt_state_dict.items()}
    ckpt_state_dict = {k.replace(".fastlora_up_proj", ".up_proj"): v for k, v in ckpt_state_dict.items()}
    ckpt_state_dict = {k.replace(".fastlora_down_proj", ".down_proj"): v for k, v in ckpt_state_dict.items()}
    # model.layers.0.self_attn.o_proj.fastlora_A1.weight -> base_model.model.model.layers.0.base_layer.self_attn.o_proj.fastlora_A1.weight
    def mapping_fn(x):
        return x.replace("model.layers", "base_model.model.model.layers").replace("self_attn", "base_layer.self_attn").replace("mlp", "base_layer.mlp")
    fastlora_state_dict = {mapping_fn(k): v for k, v in ckpt_state_dict.items()}
    # cast to torch_dtype
    fastlora_state_dict = {k: v.to(torch_dtype) for k, v in fastlora_state_dict.items()}
    del ckpt_model, ckpt_state_dict

    with open(os.path.join(model_name_or_path, "config.json"), "r") as f:
        config_dict = json.load(f)
        # HACK: we need to use the base model name or path to load the model
        if "mistral" in model_name_or_path.lower():
            base_model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
            # base_model_name_or_path = config_dict["_name_or_path"]
        else:
            raise ValueError(f"Cannot recognize base model name: {model_name_or_path}")
        config_dict = {k: v for k, v in config_dict.items() if "fastlora_" in k}
        fastlora_param_mapping = {"o": "o_proj", "up": "up_proj", "down": "down_proj"}
        config_dict["fastlora_param"] = [fastlora_param_mapping.get(x, x) for x in config_dict["fastlora_param"]]
        config_dict["peft_type"] = "FASTLORA"
        config_dict["task_type"] = "CAUSAL_LM"
    fastlora_config = FastLoraConfig(**config_dict)

    # print(fastlora_config)

    model = MistralForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
        **kwargs
    )
    model = FastLoraModelForCausalLM(model, fastlora_config)

    missing_keys = set(model.state_dict().keys()) - set(fastlora_state_dict.keys())
    missing_keys = [key for key in missing_keys if "fastlora_" in key]
    # print(f"missing_keys: {missing_keys}")
    # print(f"ckpt_state_dict.keys(): {ckpt_state_dict.keys()}")

    assert len(missing_keys) == 0, f"missing_keys: {missing_keys}"

    fastlora_state_dict, mismatched_keys = _find_mismatched_keys(
        model, fastlora_state_dict, ignore_mismatched_sizes=False,
    )
    if mismatched_keys:
        # see https://github.com/huggingface/transformers/blob/09f9f566de83eef1f13ee83b5a1bbeebde5c80c1/src/transformers/modeling_utils.py#L4039
        mismatched_warning = "\n".join(
            [
                f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                for key, shape1, shape2 in mismatched_keys
            ]
        )
        msg = (
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint "
            f"and are being ignored because you passed `ignore_mismatched_sizes=True`: {mismatched_warning}."
        )
        warnings.warn(msg)

    assert len(mismatched_keys) == 0, f"mismatched_keys: {mismatched_keys}"

    model.load_state_dict(fastlora_state_dict, strict=False)

    return model


def get_fastlora_model(model, fastlora_config):
    return FastLoraModelForCausalLM(model, fastlora_config)

if __name__ == "__main__":
    # load mistral 7B
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import peft.peft_model as peft_model
    import peft.mapping as peft_mapping
    peft_model.PEFT_TYPE_TO_MODEL_MAPPING.update({"FASTLORA": FastLoraModel})
    peft_mapping.PEFT_TYPE_TO_CONFIG_MAPPING.update({"FASTLORA": FastLoraConfig})
    peft_model.get_peft_model_state_dict = get_peft_model_state_dict
    peft_model.set_peft_model_state_dict = set_peft_model_state_dict
    

    # # Test 1:
    # model_path = "mistralai/Mistral-7B-Instruct-v0.2"
    # model = AutoModelForCausalLM.from_pretrained(model_path)
    # config = FastLoraConfig(
    #     fastlora_r=1024,
    #     fastlora_norm="svd",
    #     target_modules=["o_proj"],
    #     task_type="CAUSAL_LM",
    #     peft_type="FASTLORA",
    # )
    # model = get_fastlora_model(model, config)
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer.pad_token = tokenizer.eos_token
    # print(model)


    # # Test 2: load
    # from transformers import AutoTokenizer
    # # model_path = "data/outputs/fastlora.Mistral7BInstructv02.mistral-2K-1B-mix.w1024.kinf.r1024.a64.o.svd.copying.pretrain.20240819-224301/checkpoint-15259"
    # # model_path = "data/outputs/fastlora.Mistral7BInstructv02.mistral-8K-2B-mix.w4096.kinf.r1024.a64.o.svd.copying.pt.20240820-205609/checkpoint-11248"
    # # model_path = "data/outputs/fastlora.Mistral7BInstructv02.mistral-8K-2B-mix.w4096.kinf.r1024.a64.o.softmax.copying.pt.20240820-205639/checkpoint-14795"
    # # model_path = "data/outputs/fastlora.Mistral7BInstructv02.mistral-2K-1B-mix.w1024.kinf.r1024.a64.o.frobenius.copying.pretrain.20240819-203800/checkpoint-15259"
    # # model_path = "data/models-dev/fastlora.Mistral7BInstructv02.instruct-v2.w4096.kinf.r1024.a64.o.svd.copying.pt-sft-v2.20240904-222324/checkpoint-2688"
    # # model_path = "data/models-dev/fastlora.Mistra7BInstructv02.instruct-qa.w1024.kinf.r128.a64.o.softmax.pretrain-instruct.20240813-005206/checkpoint-2928"
    # model_path = "data/models-dev/fastlora.Mistral7BInstructv02.instruct-v1.w4096.kinf.r1024.a64.o.svd.copying.pt-sft-v1.20240904-223049/checkpoint-3108"
    # model = load_pretrained_model(model_path)
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # print(model)

    # # # print the parameters
    # for name, param in model.named_parameters():
    #     print(name, param.shape, param.dtype, param.device)
    #     print(param)
    
    model_name_or_path = "data/models-dev/fastlora.Mistral7BInstructv02.mistral-8K-1B-com.w1024-pre-norm-sum.kinf.ri1024.r128.a64.o.svd.bs8.lr1e-4.pt-mix.20240916-110738/checkpoint-14332"
    from peft.config import PeftConfig
    peft_config = PeftConfig.from_pretrained(model_name_or_path)
    base_model_path = peft_config.base_model_name_or_path
    assert base_model_path is not None, "base_model_name_or_path should not be None"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
    )
    peft_config.task_type = "FAST_LORA_CAUSAL_LM"
    peft_config.fastlora_merge = "pre-norm-sum"
    model = FastLoraModelForCausalLM.from_pretrained(
        base_model,
        model_name_or_path,
        adapter_name='default',
        is_trainable=False,
        config=peft_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


    ## Task 3: test forward
    from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer.pad_token = tokenizer.eos_token
    # inputs = tokenizer(
    #     ["Hello, my dog is cute. I love him!", "what pet do I have? a dog."], 
    #     return_tensors="pt",
    #     padding=True,
    # )
    # input_ids = inputs["input_ids"].unsqueeze(0)
    # attention_mask = inputs["attention_mask"].unsqueeze(0)
    # print('input_ids', input_ids)
    # print('attention_mask', attention_mask)
    # outputs = model(
    #     input_ids=input_ids, attention_mask=attention_mask,
    #     labels=input_ids,
    #     output_attentions=True, output_hidden_states=True,
    # )
    # print(outputs.keys())
    # print(outputs["logits"].shape)
    # print(outputs["past_key_values"][0][0].shape)
    # print(outputs["attentions"][0].shape)
    # print(outputs["hidden_states"][0].shape)


    ## Task 4: compute perplexity
    from fastlora.data import Data
    from fastlora.utils import DefaultDataCollator
    from fastlora.modeling_utils import evaluate_perplexity, compute_loss
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from tqdm import tqdm
    from torch.nn.utils.rnn import pad_sequence
    
    accelerator = Accelerator()
    model = accelerator.prepare(model)

    eval_data = "data/pretrain/val-8K-1M/data.json"
    dataset_cache_dir = "data/cache"
    eval_dataset = Data.prepare_eval_data(
        eval_data, 
        tokenizer=tokenizer,
        max_length=65536,
        min_length=512,
        window_size=65536,
        # chat_template='llama-2',
        seed=42,
        # cache_dir=dataset_cache_dir,
    )
    dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=DefaultDataCollator(tokenizer))
    dataloader = accelerator.prepare(dataloader)

    # for name, param in model.named_parameters():
    #     print(name, param.shape, param.dtype, param.device)

    # for window_size in [1024, 512, 256, 128]:
    # for window_size in [8192, 4096, 2048, 1024, 512]:
    # for target_pos in [0, 1024, 2048, 3072, 4096, 5120, 6144, 7168]:
    for _ in [None]:

        # perplexity = evaluate_perplexity(model, dataloader, accelerator)
        all_loss = []
        with torch.inference_mode():
            for i, x in enumerate(tqdm(dataloader, desc="Computing Perplexity")):

                if x["input_ids"].shape[-1] < 16384:
                    continue

                # prepare the context and the input
                context_len = 1024
                window_size = 1024
                target_pos = 0
                target_len = 1024

                assert x["input_ids"].shape[:2] == (1, 1), "batch size should be 1"
                input_ids_seq_1, input_ids_seq_2 = x["input_ids"][:, 0, :context_len], x["input_ids"][:, 0, context_len:]
                attention_mask_seq_1, attention_mask_seq_2 = x["attention_mask"][:, 0, :context_len], x["attention_mask"][:, 0, context_len:]
                label_seq_1, label_seq_2 = x["labels"][:, 0, :context_len], x["labels"][:, 0, context_len:]
                # input_ids_seq_2, attention_mask_seq_2, label_seq_2 = input_ids_seq_1, attention_mask_seq_1, label_seq_1     # for reconstruction evaluation
                label_seq_1 = torch.full_like(input_ids_seq_1, -100)

                # for short instruction evaluation
                # input_ids_seq_2, attention_mask_seq_2, label_seq_2 = input_ids_seq_2[:, :target_len], attention_mask_seq_2[:, :target_len], label_seq_2[:, :target_len]
                input_ids_seq_2, attention_mask_seq_2, label_seq_2 = input_ids_seq_2[:, target_pos:target_pos + target_len], attention_mask_seq_2[:, target_pos:target_pos + target_len], label_seq_2[:, target_pos:target_pos + target_len]
                # input_ids_seq_2, attention_mask_seq_2, label_seq_2 = input_ids_seq_2[:, -target_len:], attention_mask_seq_2[:, -target_len:], label_seq_2[:, -target_len:]
                # input_ids_seq_1, attention_mask_seq_1 = input_ids_seq_1[:, :64], attention_mask_seq_1[:, :64]     # for short context evaluation

                # cut the context into segments
                number_windows = (input_ids_seq_1.shape[-1] + window_size - 1) // window_size
                seq_len = (input_ids_seq_1.shape[-1] + number_windows - 1) // number_windows
                input_ids_seq_1 = F.pad(input_ids_seq_1, (0, number_windows * seq_len - input_ids_seq_1.shape[-1]), value=tokenizer.pad_token_id).reshape(-1, number_windows, seq_len)
                attention_mask_seq_1 = F.pad(attention_mask_seq_1, (0, number_windows * seq_len - attention_mask_seq_1.shape[-1]), value=0).reshape(-1, number_windows, seq_len)
                label_seq_1 = F.pad(label_seq_1, (0, number_windows * seq_len - label_seq_1.shape[-1]), value=-100).reshape(-1, number_windows, seq_len)
                print(input_ids_seq_1.shape, input_ids_seq_2.shape)
                print(attention_mask_seq_1.shape, attention_mask_seq_2.shape)
                print(label_seq_1.shape, label_seq_2.shape)

                # >>> Mode 1: default
                assert input_ids_seq_1.shape[0] == 1, "batch size should be 1"
                input_ids = pad_sequence([*input_ids_seq_1.squeeze(0), input_ids_seq_2.squeeze(0)], batch_first=True, padding_value=tokenizer.pad_token_id).unsqueeze(0)
                attention_mask = pad_sequence([*attention_mask_seq_1.squeeze(0), attention_mask_seq_2.squeeze(0)], batch_first=True, padding_value=0).unsqueeze(0)
                labels = pad_sequence([*label_seq_1.squeeze(0), label_seq_2.squeeze(0)], batch_first=True, padding_value=-100).unsqueeze(0)
                inputs = {}
                inputs["input_ids"] = input_ids
                inputs["attention_mask"] = attention_mask
                inputs["labels"] = labels
                # inputs["labels"] = inputs["input_ids"]
                print(inputs["input_ids"].shape, inputs["input_ids"])
                print(inputs["attention_mask"].shape, inputs["attention_mask"])
                print(inputs["labels"].shape, inputs["labels"])
                outputs = model(**inputs)
                # print(outputs.logits[0, 0, :8, :8])
                # print(outputs.logits[0, 1, :8, :8])
                logits = outputs.logits
                loss, batch_loss, valid_token_num = compute_loss(logits.reshape((-1,) + logits.shape[-2:]), labels.reshape(-1, labels.shape[-1]), shift=True)
                print(loss, batch_loss, valid_token_num)
                # all_loss.extend(batch_loss.tolist())
                all_loss.append(loss.item())
                # <<< Mode 1 <<<
                
                # >>> Mode 2: states >>>
                # input_ids_seq_1 = input_ids_seq_1.transpose(0, 1)
                # attention_mask_seq_1 = attention_mask_seq_1.transpose(0, 1)
                outputs = model(
                    input_ids=input_ids_seq_1, attention_mask=attention_mask_seq_1,
                    output_hidden_states=True,
                )
                hidden_states_seq_1 = outputs.hidden_states
                # hidden_states_seq_1 = [x.transpose(0, 1) for x in hidden_states_seq_1]
                # attention_mask_seq_1 = attention_mask_seq_1.transpose(0, 1)
                # print(hidden_states_seq_1[0].shape)
                # print(attention_mask_seq_1.shape)
                outputs = model(
                    input_ids=input_ids_seq_2, attention_mask=attention_mask_seq_2,
                    fastlora_hidden_states_and_mask=(hidden_states_seq_1, attention_mask_seq_1),
                )
                # print(outputs.loss)
                logits = outputs.logits
                labels = label_seq_2
                loss, batch_loss, valid_token_num = compute_loss(logits, labels, shift=True)
                print(loss, batch_loss, valid_token_num)
                all_loss.append(loss.item())
                # <<< Mode 2 <<<

                # >>> Mode 3: weights >>>
                outputs = model(
                    input_ids=input_ids_seq_1, attention_mask=attention_mask_seq_1,
                    output_hidden_states=True,
                )
                hidden_states_seq_1 = outputs.hidden_states
                lora_weights = model.to_lora_weights(hidden_states_seq_1, attention_mask_seq_1)
                # print(hidden_states_seq_1[0].shape)
                # print(attention_mask_seq_1.shape)
                # print(list(lora_weights.items())[0][1]["lora_a"].shape)
                outputs = model(
                    input_ids=input_ids_seq_2, attention_mask=attention_mask_seq_2,
                    fastlora_weights=lora_weights,
                )
                # print(outputs.loss)
                logits = outputs.logits
                labels = label_seq_2
                loss, batch_loss, valid_token_num = compute_loss(logits, labels, shift=True)
                print(loss, batch_loss, valid_token_num)
                all_loss.append(loss.item())
                # <<< Mode 3 <<<
                
                # # >>> Generation Mode 2 >>>
                # outputs = model(
                #     input_ids=input_ids_seq_1, attention_mask=attention_mask_seq_1,
                #     output_hidden_states=True,
                # )
                # hidden_states_seq_1 = outputs.hidden_states
                # outputs = model.generate(
                #     inputs=input_ids_seq_2,
                #     fastlora_hidden_states_and_mask=(hidden_states_seq_1, attention_mask_seq_1),
                #     max_new_tokens=20,
                # )
                # # print(outputs)
                # output_text = tokenizer.decode(outputs[0, input_ids_seq_2.shape[-1]:], skip_special_tokens=True)
                # print(output_text)
                # # <<< Generation <<<
                
                # # >>> Generation Mode 3 >>>
                # outputs = model(
                #     input_ids=input_ids_seq_1, attention_mask=attention_mask_seq_1,
                #     output_hidden_states=True,
                # )
                # hidden_states_seq_1 = outputs.hidden_states
                # lora_weights = model.to_lora_weights(hidden_states_seq_1, attention_mask_seq_1)
                # outputs = model.generate(
                #     inputs=input_ids_seq_2,
                #     fastlora_weights=lora_weights,
                #     max_new_tokens=20,
                # )
                # # print(outputs)
                # output_text = tokenizer.decode(outputs[0, input_ids_seq_2.shape[-1]:], skip_special_tokens=True)
                # print(output_text)
                # # <<< Generation <<<

                # break
        perplexity = math.exp(sum(all_loss) / len(all_loss))

        print(perplexity)


