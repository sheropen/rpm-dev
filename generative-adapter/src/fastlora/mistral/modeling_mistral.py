# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Mistral model."""
import inspect
import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.integrations import is_deepspeed_zero3_enabled
from .configuration_mistral import MistralConfig

# if is_flash_attn_2_available():
#     from flash_attn import flash_attn_func, flash_attn_varlen_func
#     from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

#     _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

from ultragist.modeling_ultragist import Memory
from ultragist.modeling_utils import optional_grad_ctx, compute_loss, ModelOutput


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MistralConfig"


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Mistral
class MistralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MistralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class MistralBatchNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MistralBatchNorm compute the mean and variance for each hidden dimension
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
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


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Mistral
class MistralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class MistralLinearScalingRotaryEmbedding(MistralRotaryEmbedding):
    """MistralRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=32768, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class MistralDynamicNTKScalingRotaryEmbedding(MistralRotaryEmbedding):
    """MistralRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=32768, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from streaming-llm
def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


# class MLP(nn.Module):
#     """
#     multi layer perceptron.
#     Args:
#         hidden_sizes: list of hidden sizes
#     """
#     def __init__(self, hidden_sizes: List[int]):
#         super().__init__()
#         self.layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(len(hidden_sizes) - 1)])

#     def _init_weights(self, mean=0.0, std=1.0, zero=False):
#         for layer in self.layers:
#             if zero:
#                 layer.weight.data.zero_()
#             else:
#                 layer.weight.data.normal_(mean, std)
#             if layer.bias is not None:
#                 layer.bias.data.zero_()

#     def forward(self, x):
#         for layer in self.layers:
#             # x = F.relu(layer(x))
#             x = layer(x)
#         return x

class FastLoraLinear(nn.Module):
    def __init__(
        self,
        base_layer,
        in_features: int,
        out_features: int,
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
    ):
        assert base_layer.in_features == in_features
        assert base_layer.out_features == out_features

        super().__init__()
        
        self.base_layer = base_layer
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size
        
        self.lora_r = lora_r
        if lora_r > 0:
            self.lora_alpha = lora_alpha
            self.lora_scaling = lora_alpha / lora_r
            self.lora_dropout = nn.Dropout(p=lora_dropout)
            self.lora_A = nn.Linear(in_features, lora_r, bias=False)
            self.lora_B = nn.Linear(lora_r, out_features, bias=False)
            self.reset_lora_parameters()

        self.fastlora_r = fastlora_r
        if fastlora_r > 0:
            self.fastlora_max_rank = fastlora_max_rank
            self.fastlora_inter_size = fastlora_inter_size if fastlora_inter_size is not None else self.fastlora_r
            self.fastlora_alpha = fastlora_alpha
            self.fastlora_scaling = fastlora_alpha / fastlora_r
            self.fastlora_arch = fastlora_arch
            self.fastlora_norm = fastlora_norm
            self.fastlora_init = fastlora_init
            self.fastlora_dropout = nn.Dropout(p=fastlora_dropout)
            if "batchnorm" in fastlora_norm:
                self.hidden_state_norm_fn = MistralBatchNorm(hidden_size)
            else:
                self.hidden_state_norm_fn = MistralRMSNorm(hidden_size)
            # if fastlora_norm == "rms":
            #     self.hidden_state_norm_fn = MistralRMSNorm(hidden_size)
            # else:
            #     self.hidden_state_norm_fn = self._rms_norm
            self.create_fastlora_parameters()
            self.reset_fastlora_parameters()


    def create_fastlora_parameters(self):
        if self.fastlora_arch == "as":
            raise NotImplementedError()
            self.fastlora_A = nn.Linear(self.in_features, self.fastlora_r, bias=False)
            self.fastlora_B = nn.Parameter(torch.ones(self.out_features))
            assert self.hidden_size == self.out_features
        elif self.fastlora_arch == "sb":
            raise NotImplementedError()
            self.fastlora_A = nn.Parameter(torch.ones(self.in_features))
            self.fastlora_B = nn.Linear(self.fastlora_r, self.out_features, bias=False)
            assert self.hidden_size == self.in_features
        elif self.fastlora_arch == "asbb":
            raise NotImplementedError()
            self.fastlora_A = nn.Linear(self.in_features, self.fastlora_r, bias=False)
            self.fastlora_B1 = nn.Linear(self.hidden_size, self.fastlora_r, bias=False)
            self.fastlora_B2 = nn.Linear(self.fastlora_r, self.out_features, bias=False)
        elif self.fastlora_arch == "aasb":
            raise NotImplementedError()
            self.fastlora_A1 = nn.Linear(self.in_features, self.fastlora_r, bias=False)
            self.fastlora_A2 = nn.Linear(self.fastlora_r, self.hidden_size, bias=False)
            self.fastlora_B = nn.Linear(self.fastlora_r, self.out_features, bias=False)
        elif self.fastlora_arch == "assb":
            raise NotImplementedError()
            self.fastlora_A = nn.Linear(self.in_features, self.fastlora_r, bias=False)
            self.fastlora_B = nn.Linear(self.fastlora_r, self.out_features, bias=False)
        elif self.fastlora_arch == "aassbb" or self.fastlora_arch == "aaab" or self.fastlora_arch == "ab":
            self.fastlora_A1 = nn.Linear(self.in_features, self.fastlora_inter_size, bias=False)
            self.fastlora_A2 = nn.Linear(self.hidden_size, self.fastlora_inter_size, bias=False)
            self.fastlora_A3 = nn.Linear(self.hidden_size, self.fastlora_r, bias=False)
            self.fastlora_B = nn.Linear(self.fastlora_r, self.out_features, bias=False)
        else:
            raise ValueError(f"Unknown fastlora_arch: {self.fastlora_arch}")
    
    def _init_prameters(self, kwargs=None):
        self.reset_lora_parameters()
        self.reset_fastlora_parameters(kwargs)

    def reset_fastlora_parameters(self, kwargs=None):
        if self.fastlora_r > 0:
            # if hidden_state_norm_fn has weight and bias, then init it
            self.hidden_state_norm_fn.reset_parameters()

            # initialize the fastlora parameters
            if self.fastlora_arch == "as":
                raise NotImplementedError()
                nn.init.kaiming_normal_(self.fastlora_A.weight, a=math.sqrt(5))
                nn.init.ones_(self.fastlora_B)
            elif self.fastlora_arch == "sb":
                raise NotImplementedError()
                nn.init.ones_(self.fastlora_A)
                nn.init.zeros_(self.fastlora_B.weight)
            elif self.fastlora_arch == "asbb":
                raise NotImplementedError()
                nn.init.kaiming_normal_(self.fastlora_A.weight, a=math.sqrt(5))
                nn.init.kaiming_normal_(self.fastlora_B1.weight, a=math.sqrt(5))
                nn.init.zeros_(self.fastlora_B2.weight)
            elif self.fastlora_arch == "aasb":
                raise NotImplementedError()
                nn.init.kaiming_normal_(self.fastlora_A1.weight, a=math.sqrt(5))
                nn.init.kaiming_normal_(self.fastlora_A2.weight, a=math.sqrt(5))
                nn.init.zeros_(self.fastlora_B.weight)
            elif self.fastlora_arch == "assb":
                raise NotImplementedError()
                nn.init.kaiming_normal_(self.fastlora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.fastlora_B.weight)
            elif self.fastlora_arch == "aassbb" or self.fastlora_arch == "aaab" or self.fastlora_arch == "ab":
                nn.init.kaiming_normal_(self.fastlora_A1.weight, mode='fan_in', a=math.sqrt(5))
                nn.init.kaiming_normal_(self.fastlora_A2.weight, mode='fan_in', a=math.sqrt(5))
                nn.init.kaiming_normal_(self.fastlora_A3.weight, mode='fan_in', a=math.sqrt(5))
                nn.init.zeros_(self.fastlora_B.weight)
                if self.fastlora_init == "copying" and kwargs is not None:
                    print("Initializing fastlora_A2 and fastlora_A3 with base model weights")
                    assert kwargs is not None, "kwargs is required for copying"
                    assert "self_attn.k_proj" in kwargs, "self_attn.k_proj is required for copying"
                    assert "self_attn.k_proj" in kwargs, "self_attn.k_proj is required for copying"
                    assert self.fastlora_A2.weight.shape == kwargs["self_attn.k_proj"].weight.shape, f"self_attn.k_proj.shape: {kwargs['self_attn.k_proj'].weight.shape}, fastlora_A2.shape: {self.fastlora_A2.weight.shape}"
                    assert self.fastlora_A3.weight.shape == kwargs["self_attn.v_proj"].weight.shape, f"self_attn.v_proj.shape: {kwargs['self_attn.v_proj'].weight.shape}, fastlora_A3.shape: {self.fastlora_A3.weight.shape}"
                    self.fastlora_A2.weight.data.copy_(kwargs["self_attn.k_proj"].weight.data)
                    self.fastlora_A3.weight.data.copy_(kwargs["self_attn.v_proj"].weight.data)
            else:
                raise ValueError(f"Unknown fastlora_arch: {self.fastlora_arch}")

    def reset_lora_parameters(self):
        if self.lora_r > 0:
            nn.init.kaiming_normal_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
    
    def _rms_norm(self, hidden_states, variance_epsilon=1e-6):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance_sum = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance_sum + variance_epsilon)
        return hidden_states.to(input_dtype)

    def _frobenius_norm(self, M, variance_epsilon=1e-6):
        # shape: ... x DI x DO
        assert M.dim() == 3 or 4
        input_dtpye = M.dtype
        DI, DO = M.shape[-2], M.shape[-1]
        # print(M)
        M = M.to(torch.float32)
        variance_sum = M.pow(2).sum(dim=(-1, -2), keepdim=True) / DO
        M = M * torch.rsqrt(variance_sum + variance_epsilon)
        # print(DI, DO, variance_sum.mean())
        return M.to(input_dtpye)
    
    def _spectral_norm(self, M, epsilon=1e-6):
        # shape: ... x DI x DO
        assert M.dim() == 3 or 4
        input_dtpye = M.dtype
        DI, DO = M.shape[-2], M.shape[-1]
        M = M.to(torch.float32)
        S = torch.linalg.svdvals(M)    # ... x min(DI, DO)
        # print(S.shape, S.mean(), S.max(), S.min())
        max_s = S[..., 0]   # ... x 1
        # print("spectral", max_s)
        # print("spectral 2 norm", torch.sqrt(S.pow(2).sum(dim=-1)))
        # print("frobenius", torch.sqrt(M.pow(2).sum(dim=(-1, -2))))
        M = M / (max_s.unsqueeze(-1).unsqueeze(-1) / DO + epsilon)
        return M.to(input_dtpye)

    def _svd_norm(self, M):
        # shape: B x DI x DO
        assert M.dim() == 3 
        input_dtpye = M.dtype
        M = M.to(torch.float32)
        if min(M.shape[-2], M.shape[-1]) <= self.fastlora_max_rank or (not self.training):
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            U = U[..., :self.fastlora_max_rank]
            S = S[..., :self.fastlora_max_rank]
            V = Vh.transpose(-2, -1)
            V = V[..., :self.fastlora_max_rank]
            # print(S.mean(), S.max(), S.min())
            # print(S[:, :128].mean(), S[:, :128].sort(descending=True).values[0, :4], S[:, :128].sort(descending=True).values[0, -4:])
        else:
            U, S, V = torch.svd_lowrank(M, q=self.fastlora_max_rank, niter=1)
            # print(S.mean(), S.max(), S.min())
            # print(S[:, :128].mean(), S[:, :128].sort(descending=True).values[0, :4], S[:, :128].sort(descending=True).values[0, -4:])
        Vh = V.transpose(-2, -1)
        M_norm = U @ Vh
        # print(M.shape, M_norm.shape, U.shape, S.shape, Vh.shape)
        return M_norm.to(input_dtpye)

    # def _svd_lstsq(self, A, B, tol=1e-2):
    #     # A shape: ... x m x n
    #     # B shape: ... x m x p
    #     assert A.dtype == B.dtype, f"A.dtype: {A.dtype}, B.dtype: {B.dtype}"
    #     assert A.dim() == 3 and B.dim() == 3, f"A.dim(): {A.dim()}, B.dim(): {B.dim()}"
    #     input_dtpye = A.dtype
    #     A = A.to(torch.float32)
    #     B = B.to(torch.float32)
    #     U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    #     assert S.dim() == 2 and S.shape[0] == 1, f"only support batch size 1, but got {S.shape}"
    #     rank = (S > tol).sum()
    #     U, S, Vh = U[..., :, :rank], S[..., :rank], Vh[..., :rank, :]
    #     print('S', S.shape, S.mean(), S.max(), S.min())
    #     Sinv = 1 / S
    #     UhB = U.transpose(-2, -1) @ B   # ... x rank x p
    #     Sinv = Sinv.unsqueeze(-1)
    #     SinvUhB = Sinv * UhB    # ... x rank x p
    #     output = Vh.transpose(-2, -1) @ SinvUhB   # ... x n x p
    #     return output.to(input_dtpye)

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


    def forward(self, x: torch.Tensor, past_hidden_states: Optional[torch.Tensor] = None):
        # print(past_hidden_states.shape if past_hidden_states is not None else None)
        x_input = x
        result = self.base_layer(x_input)
        if self.lora_r > 0:
            result = result + self.lora_B(self.lora_A(self.lora_dropout(x_input))) * self.lora_scaling
        if self.fastlora_r > 0 and past_hidden_states is not None:
            assert past_hidden_states.dim() == 3    # B, R, D
            past_hidden_states_norm = self.hidden_state_norm_fn(past_hidden_states)
            x = self.fastlora_dropout(x_input)
            if self.fastlora_arch == "as":
                raise NotImplementedError()
                x = self.fastlora_A * x
                x = x.bmm(past_hidden_states)
                x = x.dot(self.fastlora_B)
                result = result + self.fastlora_scaling * x
            elif self.fastlora_arch == "sb":
                raise NotImplementedError()
                x = x.dot(self.fastlora_A)
                x = x.bmm(past_hidden_states.transpose(1, 2))
                x = self.fastlora_B * x
                result = result + self.fastlora_scaling * x
            elif self.fastlora_arch == "asbb":
                raise NotImplementedError()
                x = self.fastlora_A(x)
                x = x.bmm(past_hidden_states)
                x = self.fastlora_A3(x)
                x = self.fastlora_B(x)
                result = result + self.fastlora_scaling * x
            elif self.fastlora_arch == "aasb":
                raise NotImplementedError()
                x = self.fastlora_A1(x)
                x = self.fastlora_A2(x)
                x = x.bmm(past_hidden_states.transpose(1, 2))
                x = self.fastlora_B(x)
                result = result + self.fastlora_scaling * x
            elif self.fastlora_arch == "assb":
                raise NotImplementedError()
                x = self.fastlora_A(x)
                x = x.bmm(past_hidden_states)
                x = x.bmm(past_hidden_states.transpose(1, 2))
                x = self.fastlora_B(x)
                result = result + self.fastlora_scaling * x
            elif self.fastlora_arch == "aassbb":
                if self.fastlora_norm == "xnorm":
                    M = past_hidden_states_norm.shape[1]
                    R = self.fastlora_inter_size
                    x_a = self.fastlora_A1(x)     # B, N, RI
                    A2_hidden_states = self.fastlora_A2(past_hidden_states_norm)   # B, M, RI
                    x_aas = torch.bmm(x_a, A2_hidden_states.transpose(1, 2)) / math.sqrt(R)   # B, N, M
                    variance_sum = x_aas.pow(2).sum(dim=-1, keepdim=True)
                    x_aas_norm = x_aas * torch.rsqrt(variance_sum + 1e-6)
                    A3_hidden_states = self.fastlora_A3(past_hidden_states_norm)    # B, M, R
                    x_aassa = torch.bmm(x_aas_norm, A3_hidden_states) / math.sqrt(R)    # B, N, R
                    x_aassab = self.fastlora_B(x_aassa)    # B, N, D
                    result = result + self.fastlora_scaling * x_aassab
                    # print(f"x_input.norm: {x_input.norm(dim=-1).mean()} ({x_input.norm(dim=-1).max()}), x_a.norm: {x_a.norm(dim=-1).mean()} ({x_a.norm(dim=-1).max()}), x_aas.norm: {x_aas.norm(dim=-1).mean()} ({x_aas.norm(dim=-1).max()}), x_aas_norm.norm: {x_aas_norm.norm(dim=-1).mean()} ({x_aas_norm.norm(dim=-1).max()}), x_aassa.norm: {x_aassa.norm(dim=-1).mean()} ({x_aassa.norm(dim=-1).max()}), x_aassab.norm: {x_aassab.norm(dim=-1).mean()} ({x_aassab.norm(dim=-1).max()}), result.norm: {result.norm(dim=-1).mean()} ({result.norm(dim=-1).max()})")
                elif self.fastlora_norm == "softmax":
                    x_query_states = self.fastlora_A1(x)     # B, N, RI
                    c_key_states = self.fastlora_A2(past_hidden_states_norm)   # B, M, RI
                    c_value_states = self.fastlora_A3(past_hidden_states_norm)    # B, M, R
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        x_query_states, c_key_states, c_value_states,
                        attn_mask=None, dropout_p=0.0, is_causal=False,
                    )   # B, N, R
                    x = self.fastlora_B(attn_output)    # B, N, D
                    result = result + self.fastlora_scaling * x
                elif self.fastlora_norm == "softmax-multi":
                    num_head = 8
                    B, N, _ = x.shape
                    B, M, _ = past_hidden_states_norm.shape
                    x_query_states = self.fastlora_A1(x)     # B, N, R
                    c_key_states = self.fastlora_A2(past_hidden_states_norm)   # B, M, R
                    c_value_states = self.fastlora_A3(past_hidden_states_norm)    # B, M, R
                    x_query_states = x_query_states.view(B, N, num_head, -1).transpose(1, 2).contiguous()    # B, H, N, RR
                    c_key_states = c_key_states.view(B, M, num_head, -1).transpose(1, 2).contiguous()   # B, H, M, RR
                    c_value_states = c_value_states.view(B, M, num_head, -1).transpose(1, 2).contiguous()    # B, H, M, RR
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        x_query_states, c_key_states, c_value_states,
                        attn_mask=None, dropout_p=0.0, is_causal=False,
                    )   # B, N, R
                    attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, -1)    # B, N, R
                    x = self.fastlora_B(attn_output)    # B, N, D
                    result = result + self.fastlora_scaling * x
                elif self.fastlora_norm == "frobenius-multi":
                    num_head = 8
                    B, N, _ = x.shape
                    B, M, _ = past_hidden_states_norm.shape
                    x_a = self.fastlora_A1(x)     # B, N, R
                    A2_hidden_states = self.fastlora_A2(past_hidden_states_norm)   # B, M, R
                    A3_hidden_states = self.fastlora_A3(past_hidden_states_norm)    # B, M, R
                    x_a = x_a.view(B, N, num_head, -1).transpose(1, 2).contiguous()    # B, H, N, RR
                    A2_hidden_states = A2_hidden_states.view(B, M, num_head, -1).transpose(1, 2).contiguous()    # B, H, M, RR
                    A3_hidden_states = A3_hidden_states.view(B, M, num_head, -1).transpose(1, 2).contiguous()    # B, H, M, RR
                    ss = (A2_hidden_states.transpose(-1, -2) @ A3_hidden_states)  / math.sqrt(M)    # B, H, RR, RR
                    ss = self._frobenius_norm(ss)   # B, H, RR, RR
                    x_aassa = x_a @ ss    # B, H, N, RR
                    x_aassa = x_aassa.transpose(1, 2).contiguous().view(B, N, -1)    # B, N, R
                    x_aassab = self.fastlora_B(x_aassa)    # B, N, D
                    result = result + self.fastlora_scaling * x_aassab
                elif self.fastlora_norm == "elu" or self.fastlora_norm == "frobenius-elu":
                    M = past_hidden_states_norm.shape[1]
                    x_a = self.fastlora_A1(x)     # B, N, RI
                    A2_hidden_states = self.fastlora_A2(past_hidden_states_norm)   # B, M, RI
                    A3_hidden_states = self.fastlora_A3(past_hidden_states_norm)    # B, M, R
                    A2_hidden_states = F.elu(A2_hidden_states) + 1
                    if self.fastlora_norm == "elu":
                        # A2: column-wise normalization
                        A2_hidden_states = A2_hidden_states / A2_hidden_states.sum(dim=-2, keepdim=True) * math.sqrt(M)
                    ss = torch.bmm(A2_hidden_states.transpose(1, 2), A3_hidden_states)  / math.sqrt(M)    # B, RI, R
                    if self.fastlora_norm == "frobenius-elu":
                        ss = self._frobenius_norm(ss)
                    x_aassa = torch.bmm(x_a, ss)    # B, N, R
                    x_aassab = self.fastlora_B(x_aassa)    # B, N, D
                    result = result + self.fastlora_scaling * x_aassab
                elif self.fastlora_norm == "least-square":
                    M = past_hidden_states_norm.shape[1]
                    x_a = self.fastlora_A1(x)     # B, N, RI
                    # print(f"x_a.norm: {x_a.norm(dim=-1).mean()} ({x_a.norm(dim=-1).max()})")
                    A2_hidden_states = self.fastlora_A2(past_hidden_states_norm)   # B, M, RI
                    A3_hidden_states = self.fastlora_A3(past_hidden_states_norm)    # B, M, R
                    ss = self._lstsq(A2_hidden_states, A3_hidden_states)
                    x_aassa = torch.bmm(x_a, ss)    # B, N, R
                    # print(f"x_aassa.norm: {x_aassa.norm(dim=-1).mean()} ({x_aassa.norm(dim=-1).max()})")
                    x_aassab = self.fastlora_B(x_aassa)    # B, N, D
                    result = result + self.fastlora_scaling * x_aassab
                else:
                    M = past_hidden_states_norm.shape[1]
                    # import time
                    # begin_time = time.time()
                    x_a = self.fastlora_A1(x)     # B, N, RI
                    # ckpt1_time = time.time()
                    A2_hidden_states = self.fastlora_A2(past_hidden_states_norm)   # B, M, RI
                    # ckpt2_time = time.time()
                    A3_hidden_states = self.fastlora_A3(past_hidden_states_norm)    # B, M, R
                    # ckpt3_time = time.time()
                    ss = torch.bmm(A2_hidden_states.transpose(1, 2), A3_hidden_states)  / math.sqrt(M)    # B, RI, R
                    # ckpt4_time = time.time()
                    if self.fastlora_norm == "frobenius" or self.fastlora_norm == "frobenius-batchnorm":
                        ss = self._frobenius_norm(ss)
                    elif self.fastlora_norm == "spectral":
                        ss = self._spectral_norm(ss)
                    elif self.fastlora_norm == "svd":
                        ss = self._svd_norm(ss)
                    # ckpt5_time = time.time()
                    x_aassa = torch.bmm(x_a, ss)    # B, N, R
                    # ckpt6_time = time.time()
                    # print(f"xa.norm: {x_a.norm(dim=-1).mean()} ({x_a.norm(dim=-1).max()}), x_aassa.norm: {x_aassa.norm(dim=-1).mean()} ({x_aassa.norm(dim=-1).max()})")
                    # _ss_svdvals = torch.linalg.svdvals(ss.to(torch.float32))
                    # print(f"ss_svdvals: {_ss_svdvals.mean()} ({_ss_svdvals[:, :4]}) ({_ss_svdvals[:, -4:]})")
                    x_aassab = self.fastlora_B(x_aassa)    # B, N, D
                    # ckpt7_time = time.time()
                    result = result + self.fastlora_scaling * x_aassab
                    # print(
                    #     f"A1: {ckpt1_time - begin_time}"
                    #     f", A2: {ckpt2_time - ckpt1_time}"
                    #     f", A3: {ckpt3_time - ckpt2_time}"
                    #     f", ss: {ckpt4_time - ckpt3_time}"
                    #     f", norm: {ckpt5_time - ckpt4_time}"
                    #     f", x_aassa: {ckpt6_time - ckpt5_time}"
                    #     f", x_aassab: {ckpt7_time - ckpt6_time}"
                    # )
                    # print(f"x_input.norm: {x_input.norm(dim=-1).mean()} ({x_input.norm(dim=-1).max()}), x_a.norm: {x_a.norm(dim=-1).mean()} ({x_a.norm(dim=-1).max()}), x_aassa.norm: {x_aassa.norm(dim=-1).mean()} ({x_aassa.norm(dim=-1).max()}), x_aassab.norm: {x_aassab.norm(dim=-1).mean()} ({x_aassab.norm(dim=-1).max()}), result.norm: {result.norm(dim=-1).mean()} ({result.norm(dim=-1).max()})")
            elif self.fastlora_arch == "aaab":
                # print(past_hidden_states_norm.shape)
                # print(x.shape)
                x = torch.bmm(x, self.fastlora_A1.weight.transpose(0, 1).unsqueeze(0))    # B, N, R
                # print(x.shape)
                x = torch.bmm(x, self.fastlora_A2.weight.unsqueeze(0))   # B, N, D
                # print(x.shape)
                x = torch.bmm(x, self.fastlora_A3.weight.transpose(0, 1).unsqueeze(0))    # B, N, R
                # print(x.shape)
                x = torch.bmm(x, self.fastlora_B.weight.transpose(0, 1).unsqueeze(0))  # B, N, D
                # print(x.shape)
                result = result + self.fastlora_scaling * x
            elif self.fastlora_arch == "ab":
                x = torch.bmm(x, self.fastlora_A1.weight.transpose(0, 1).unsqueeze(0))    # B, N, R
                x = torch.bmm(x, self.fastlora_B.weight.transpose(0, 1).unsqueeze(0))  # B, N, D
                result = result + self.fastlora_scaling * x
            else:
                raise ValueError(f"Unknown fastlora_arch: {self.fastlora_arch}")
        # print(f"x.shape: {x.shape}, result.shape: {result.shape}")
        return result


class MistralMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        self.fastlora_gate_proj = FastLoraLinear(
            self.gate_proj, in_features=self.hidden_size, out_features=self.intermediate_size, hidden_size=self.hidden_size,
            lora_r=config.lora_r if "gate" in config.lora_param else 0, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
            fastlora_r=config.fastlora_r if "gate" in config.fastlora_param else 0, fastlora_max_rank=config.fastlora_max_rank,
            fastlora_inter_size=config.fastlora_inter_size, fastlora_alpha=config.fastlora_alpha, 
            fastlora_dropout=config.fastlora_dropout, fastlora_arch=config.fastlora_arch, fastlora_norm=config.fastlora_norm,
            fastlora_init=config.fastlora_init,
        )
        self.fastlora_up_proj = FastLoraLinear(
            self.up_proj, in_features=self.hidden_size, out_features=self.intermediate_size, hidden_size=self.hidden_size,
            lora_r=config.lora_r if "up" in config.lora_param else 0, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
            fastlora_r=config.fastlora_r if "up" in config.fastlora_param else 0, fastlora_max_rank=config.fastlora_max_rank,
            fastlora_inter_size=config.fastlora_inter_size, fastlora_alpha=config.fastlora_alpha, 
            fastlora_dropout=config.fastlora_dropout, fastlora_arch=config.fastlora_arch, fastlora_norm=config.fastlora_norm,
            fastlora_init=config.fastlora_init,
        )
        self.fastlora_down_proj = FastLoraLinear(
            self.down_proj, in_features=self.intermediate_size, out_features=self.hidden_size, hidden_size=self.hidden_size,
            lora_r=config.lora_r if "down" in config.lora_param else 0, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
            fastlora_r=config.fastlora_r if "down" in config.fastlora_param else 0, fastlora_max_rank=config.fastlora_max_rank,
            fastlora_inter_size=config.fastlora_inter_size, fastlora_alpha=config.fastlora_alpha, 
            fastlora_dropout=config.fastlora_dropout, fastlora_arch=config.fastlora_arch, fastlora_norm=config.fastlora_norm,
            fastlora_init=config.fastlora_init,
        )

        # if "mlp" in config.ultragist_param:            
        #     self.ultragist_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        #     self.ultragist_up_proj.weight.data.zero_()
        #     self.ultragist_up_proj._is_hf_initialized = True

        #     self.ultragist_down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        #     self.ultragist_down_proj.weight.data.zero_()
        #     self.ultragist_down_proj._is_hf_initialized = True
    

    def _init_fastlora_proj(self, missing_keys, kwargs=None):
    #     """Initialize the ultragist projection weight with that of the ordinal projection."""
    #     if "mlp" in self.config.ultragist_param:
    #         if is_deepspeed_zero3_enabled():
    #             import deepspeed
    #             params = [self.up_proj.weight, self.down_proj.weight, self.ultragist_up_proj.weight, self.ultragist_down_proj.weight]
    #             with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
    #                 if (self.ultragist_up_proj.weight.sum(-1) == 0).any():
    #                     self.ultragist_up_proj.weight.data[:] = self.up_proj.weight.data
    #                     self.ultragist_down_proj.weight.data[:] = self.down_proj.weight.data
    #         else:
    #             if any("ultragist_up_proj" in missing_key for missing_key in missing_keys):
    #                 # only copy the value in-place, without tieing the weight
    #                 self.ultragist_up_proj.weight.data[:] = self.up_proj.weight.data
    #                 self.ultragist_down_proj.weight.data[:] = self.down_proj.weight.data
        if is_deepspeed_zero3_enabled():
            import deepspeed
            raise NotImplementedError("FastLoRA initialization with DeepSpeed is not supported yet.")
        else:
            if any("fastlora_up_proj" in missing_key for missing_key in missing_keys):
                self.fastlora_up_proj._init_prameters(kwargs)
            if any("fastlora_down_proj" in missing_key for missing_key in missing_keys):
                self.fastlora_down_proj._init_prameters(kwargs)
            if any("fastlora_gate_proj" in missing_key for missing_key in missing_keys):
                self.fastlora_gate_proj._init_prameters(kwargs)

    def forward(self, hidden_states: torch.Tensor, past_key_value: Optional[torch.Tensor] = None, past_hidden_states: Optional[torch.Tensor] = None):
        # if "mlp" in self.config.ultragist_param:
        #     if ultragist_size > 0:
        #         ordinal_hidden_states = x[:, :-ultragist_size]
        #         ultragist_hidden_states = x[:, -ultragist_size:]

        #         ordinal_down_proj = self.down_proj(self.act_fn(self.gate_proj(ordinal_hidden_states)) * self.up_proj(ordinal_hidden_states))
        #         ultragist_down_proj = self.ultragist_down_proj(self.act_fn(self.gate_proj(ultragist_hidden_states)) * self.ultragist_up_proj(ultragist_hidden_states))
        #         down_proj = torch.cat([ordinal_down_proj, ultragist_down_proj], dim=1)
        #     else:
        #         down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        # else:
        #     down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        # return down_proj
        # result = self.fastlora_down_proj(self.act_fn(self.fastlora_gate_proj(x)) * self.fastlora_up_proj(x))
        # past_key, past_value, past_hidden_states, fastlora_rank = past_key_value
        # past_key, past_value = past_key_value

        up_proj = self.fastlora_up_proj(hidden_states, past_hidden_states=past_hidden_states)
        gate_proj = self.fastlora_gate_proj(hidden_states, past_hidden_states=past_hidden_states)
        x = self.act_fn(gate_proj) * up_proj
        result = self.fastlora_down_proj(x, past_hidden_states=past_hidden_states)
        return result


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MistralAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self._init_rope()

        self.fastlora_q_proj = FastLoraLinear(
            self.q_proj, in_features=self.hidden_size, out_features=self.num_heads * self.head_dim, hidden_size=self.hidden_size,
            lora_r=config.lora_r if "q" in config.lora_param else 0, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
            fastlora_r=config.fastlora_r if "q" in config.fastlora_param else 0, fastlora_max_rank=config.fastlora_max_rank,
            fastlora_inter_size=config.fastlora_inter_size, fastlora_alpha=config.fastlora_alpha,
            fastlora_dropout=config.fastlora_dropout, fastlora_arch=config.fastlora_arch, fastlora_norm=config.fastlora_norm,
            fastlora_init=config.fastlora_init,
        )
        self.fastlora_k_proj = FastLoraLinear(
            self.k_proj, in_features=self.hidden_size, out_features=self.num_key_value_heads * self.head_dim, hidden_size=self.hidden_size,
            lora_r=config.lora_r if "k" in config.lora_param else 0, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
            fastlora_r=config.fastlora_r if "k" in config.fastlora_param else 0, fastlora_max_rank=config.fastlora_max_rank,
            fastlora_inter_size=config.fastlora_inter_size, fastlora_alpha=config.fastlora_alpha,
            fastlora_dropout=config.fastlora_dropout, fastlora_arch=config.fastlora_arch, fastlora_norm=config.fastlora_norm,
            fastlora_init=config.fastlora_init,
        )
        self.fastlora_v_proj = FastLoraLinear(
            self.v_proj, in_features=self.hidden_size, out_features=self.num_key_value_heads * self.head_dim, hidden_size=self.hidden_size,
            lora_r=config.lora_r if "v" in config.lora_param else 0, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
            fastlora_r=config.fastlora_r if "v" in config.fastlora_param else 0, fastlora_max_rank=config.fastlora_max_rank,
            fastlora_inter_size=config.fastlora_inter_size, fastlora_alpha=config.fastlora_alpha,
            fastlora_dropout=config.fastlora_dropout, fastlora_arch=config.fastlora_arch, fastlora_norm=config.fastlora_norm,
            fastlora_init=config.fastlora_init,
        )
        self.fastlora_o_proj = FastLoraLinear(
            self.o_proj, in_features=self.num_heads * self.head_dim, out_features=self.hidden_size, hidden_size=self.hidden_size,
            lora_r=config.lora_r if "o" in config.lora_param else 0, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
            fastlora_r=config.fastlora_r if "o" in config.fastlora_param else 0, fastlora_max_rank=config.fastlora_max_rank,
            fastlora_inter_size=config.fastlora_inter_size, fastlora_alpha=config.fastlora_alpha,
            fastlora_dropout=config.fastlora_dropout, fastlora_arch=config.fastlora_arch, fastlora_norm=config.fastlora_norm,
            fastlora_init=config.fastlora_init,
        )

        # # NOTE: add extra parameters for ultragist tokens
        # # skip post initialization to speed up loading
        # if "q" in config.ultragist_param:
        #     self.ultragist_q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        #     # NOTE: initialize the ultragist parameters as zero
        #     self.ultragist_q_proj.weight.data.zero_()
        #     self.ultragist_q_proj._is_hf_initialized = True
        # if "k" in config.ultragist_param:
        #     self.ultragist_k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        #     self.ultragist_k_proj.weight.data.zero_()
        #     self.ultragist_k_proj._is_hf_initialized = True
        # if "v" in config.ultragist_param:
        #     self.ultragist_v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        #     self.ultragist_v_proj.weight.data.zero_()
        #     self.ultragist_v_proj._is_hf_initialized = True
        # if "o" in config.ultragist_param:
        #     self.ultragist_o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        #     self.ultragist_o_proj.weight.data.zero_()
        #     self.ultragist_o_proj._is_hf_initialized = True


    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = MistralRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = MistralLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = MistralDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    
    def _init_fastlora_proj(self, missing_keys, kwargs=None):
        # initialize the fastlora projection weight with gaussian noise
        # fastlora_param = self.config.fastlora_param
        if is_deepspeed_zero3_enabled():
            import deepspeed
            raise NotImplementedError("FastLoRA initialization with DeepSpeed is not supported yet.")
        else:
            if any("fastlora_q_proj" in missing_key for missing_key in missing_keys):
                self.fastlora_q_proj._init_prameters(kwargs)
            if any("fastlora_k_proj" in missing_key for missing_key in missing_keys):
                self.fastlora_k_proj._init_prameters(kwargs)
            if any("fastlora_v_proj" in missing_key for missing_key in missing_keys):
                self.fastlora_v_proj._init_prameters(kwargs)
            if any("fastlora_o_proj" in missing_key for missing_key in missing_keys):
                self.fastlora_o_proj._init_prameters(kwargs)


    # def _init_ultragist_proj(self, missing_keys):
    #     """Initialize the ultragist projection weight with that of the ordinal projection."""
    #     ultragist_param = self.config.ultragist_param
        
    #     if is_deepspeed_zero3_enabled():
    #         import deepspeed
    #         if "q" in ultragist_param:
    #             with deepspeed.zero.GatheredParameters([self.ultragist_q_proj.weight, self.q_proj.weight], modifier_rank=0):
    #                 # FIXME: after deepspeed initialization, some weights becomes non-zero
    #                 # For Llama, there are rows that are full of zeros
    #                 # For Mistral, there are values bigger than 1e29...
    #                 if (self.ultragist_q_proj.weight.sum(-1) == 0).any() or (self.ultragist_q_proj.weight > 1e29).any():
    #                     self.ultragist_q_proj.weight.data[:] = self.q_proj.weight.data
    #         if "k" in ultragist_param:
    #             with deepspeed.zero.GatheredParameters([self.ultragist_k_proj.weight, self.k_proj.weight], modifier_rank=0):
    #                 if (self.ultragist_k_proj.weight.sum(-1) == 0).any() or (self.ultragist_k_proj.weight > 1e29).any():
    #                     self.ultragist_k_proj.weight.data[:] = self.k_proj.weight.data
    #         if "v" in ultragist_param:
    #             with deepspeed.zero.GatheredParameters([self.ultragist_v_proj.weight, self.v_proj.weight], modifier_rank=0):
    #                 if (self.ultragist_v_proj.weight.sum(-1) == 0).any() or (self.ultragist_v_proj.weight > 1e29).any():
    #                     self.ultragist_v_proj.weight.data[:] = self.v_proj.weight.data
    #         if "o" in ultragist_param:
    #             with deepspeed.zero.GatheredParameters([self.ultragist_o_proj.weight, self.o_proj.weight], modifier_rank=0):
    #                 if (self.ultragist_o_proj.weight.sum(-1) == 0).any() or (self.ultragist_o_proj.weight > 1e29).any():
    #                     self.ultragist_o_proj.weight.data[:] = self.o_proj.weight.data
    #     else:
    #         # only copy the value in-place, without tieing the weight
    #         if "q" in ultragist_param and any("ultragist_q_proj" in missing_key for missing_key in missing_keys):
    #             # FIXME: some ultragist weights are not initialized as zero for mistral model, why? 
    #             # if (self.ultragist_q_proj.weight == 0).all():
    #                 self.ultragist_q_proj.weight.data[:] = self.q_proj.weight.data
    #         if "k" in ultragist_param and any("ultragist_k_proj" in missing_key for missing_key in missing_keys):
    #             # if (self.ultragist_k_proj.weight == 0).all():
    #                 self.ultragist_k_proj.weight.data[:] = self.k_proj.weight.data
    #         if "v" in ultragist_param and any("ultragist_v_proj" in missing_key for missing_key in missing_keys):
    #             # if (self.ultragist_v_proj.weight == 0).all():
    #                 self.ultragist_v_proj.weight.data[:] = self.v_proj.weight.data
    #         if "o" in ultragist_param and any("ultragist_o_proj" in missing_key for missing_key in missing_keys):
    #             # if (self.ultragist_o_proj.weight == 0).all():
    #                 self.ultragist_o_proj.weight.data[:] = self.o_proj.weight.data

            # debug
            # assert (self.ultragist_q_proj.weight.data == self.q_proj.weight.data).all()
            # assert (self.ultragist_k_proj.weight.data == self.k_proj.weight.data).all()
            # assert (self.ultragist_v_proj.weight.data == self.v_proj.weight.data).all()
            # assert (self.ultragist_o_proj.weight.data == self.o_proj.weight.data).all()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # def qkv_proj_with_fastlora(self, hidden_states, past_hidden_states=None):
    #     # past_hidden_states: (B, S, D)
    #     query_states = self.q_proj(hidden_states)
    #     if "q" in self.config.fastlora_param and past_hidden_states is not None:
    #         q_proj_a = self.fastlora_q_proj_a(past_hidden_states)  # (B, R, DI)
    #         q_proj_b = self.fastlora_q_proj_b(past_hidden_states)    # (B, R, DO)
    #         query_states_lora = torch.einsum("brd,bsd->bsr", q_proj_a, hidden_states)
    #         query_states_lora = torch.einsum("brd,bsr->bsd", q_proj_b, query_states_lora)
    #         query_states = query_states + self.config.fastlora_alpha / self.config.fastlora_rank * query_states_lora
    #     key_states = self.k_proj(hidden_states)
    #     if "k" in self.config.fastlora_param and past_hidden_states is not None:
    #         k_proj_a = self.fastlora_k_proj_a(past_hidden_states)
    #         k_proj_b = self.fastlora_k_proj_b(past_hidden_states)
    #         key_states_lora = torch.einsum("brd,bsd->bsr", k_proj_a, hidden_states)
    #         key_states_lora = torch.einsum("brd,bsr->bsd", k_proj_b, key_states_lora)
    #         key_states = key_states + self.config.fastlora_alpha / self.config.fastlora_rank * key_states_lora
    #     value_states = self.v_proj(hidden_states)
    #     if "v" in self.config.fastlora_param and past_hidden_states is not None:
    #         v_proj_a = self.fastlora_v_proj_a(past_hidden_states)
    #         v_proj_b = self.fastlora_v_proj_b(past_hidden_states)
    #         value_states_lora = torch.einsum("brd,bsd->bsr", v_proj_a, hidden_states)
    #         value_states_lora = torch.einsum("brd,bsr->bsd", v_proj_b, value_states_lora)
    #         value_states = value_states + self.config.fastlora_alpha / self.config.fastlora_rank * value_states_lora
    #     return query_states, key_states, value_states

    # def qkv_proj_with_ultragist(self, hidden_states, ultragist_size=0):
    #     if ultragist_size > 0:
    #         ordinal_hidden_states = hidden_states[:, :-ultragist_size]
    #         ultragist_hidden_states = hidden_states[:, -ultragist_size:]
            
    #         if "q" in self.config.ultragist_param:
    #             ordinal_query_states = self.q_proj(ordinal_hidden_states)
    #             ultragist_query_states = self.ultragist_q_proj(ultragist_hidden_states)
    #             query_states = torch.cat([ordinal_query_states, ultragist_query_states], dim=1)
    #         else:
    #             query_states = self.q_proj(hidden_states)

    #         if "k" in self.config.ultragist_param:
    #             ordinal_key_states = self.k_proj(ordinal_hidden_states)
    #             ultragist_key_states = self.ultragist_k_proj(ultragist_hidden_states)
    #             key_states = torch.cat([ordinal_key_states, ultragist_key_states], dim=1)
    #         else:
    #             key_states = self.k_proj(hidden_states)
            
    #         if "v" in self.config.ultragist_param:
    #             ordinal_value_states = self.v_proj(ordinal_hidden_states)
    #             ultragist_value_states = self.ultragist_v_proj(ultragist_hidden_states)
    #             value_states = torch.cat([ordinal_value_states, ultragist_value_states], dim=1)
    #         else:
    #             value_states = self.v_proj(hidden_states)

    #     else:
    #         query_states = self.q_proj(hidden_states)
    #         key_states = self.k_proj(hidden_states)
    #         value_states = self.v_proj(hidden_states)

    #     return query_states, key_states, value_states
    
    # def o_proj_with_fastlora(self, attn_output, past_hidden_states=None):
    #     attn_output_o_proj = self.o_proj(attn_output)
    #     if "o" in self.config.fastlora_param and past_hidden_states is not None:
    #         o_proj_a = self.fastlora_o_proj_a(past_hidden_states)
    #         o_proj_b = self.fastlora_o_proj_b(past_hidden_states)
    #         attn_output_lora = torch.einsum("brd,bsd->bsr", o_proj_a, attn_output)
    #         attn_output_lora = torch.einsum("brd,bsr->bsd", o_proj_b, attn_output_lora)
    #         attn_output_o_proj = attn_output_o_proj + self.config.fastlora_alpha / self.config.fastlora_rank * attn_output_lora
    #     return attn_output_o_proj

    # def o_proj_with_ultragist(self, attn_output, ultragist_size=0):
    #     if ultragist_size > 0:
    #         if "o" in self.config.ultragist_param:
    #             ordinal_attn_output = self.o_proj(attn_output[:, :-ultragist_size])
    #             ultragist_attn_output = self.ultragist_o_proj(attn_output[:, -ultragist_size:])
    #             attn_output = torch.cat([ordinal_attn_output, ultragist_attn_output], dim=1)
    #         else:
    #             attn_output = self.o_proj(attn_output)
    #     else:
    #         attn_output = self.o_proj(attn_output)
    #     return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        past_hidden_states: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()
        kv_seq_len = hidden_states.shape[-2]
        # past_key, past_value, ultragist_sizes, total_ultragist_size, raw_size_to_cache, window_size = past_key_value
        # past_key, past_value, past_hidden_states, fastlora_rank = past_key_value
        past_key, past_value = past_key_value

        if past_key is not None:
            past_seq_len = past_key.shape[2]
            kv_seq_len += past_seq_len
        else:
            past_seq_len = 0

        # query_states, key_states, value_states = self.qkv_proj_with_fastlora(hidden_states, past_hidden_states=past_hidden_states)

        query_states = self.fastlora_q_proj(hidden_states, past_hidden_states=past_hidden_states)
        key_states = self.fastlora_k_proj(hidden_states, past_hidden_states=past_hidden_states)
        value_states = self.fastlora_v_proj(hidden_states, past_hidden_states=past_hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # return keys and values before rope
        # NOTE: incrementally return keys and values for efficiency 
        # if window_size > 0:
        #     past_key_value = (key_states, value_states, ultragist_sizes, total_ultragist_size, raw_size_to_cache, window_size)

        if past_key is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        # NOTE: window_size == 0 indicates the ultragist is disabled, the model works as is, so the new past_key_values should concatenate old ones
        # if window_size == 0:
        #     past_key_value = (key_states, value_states, ultragist_sizes, total_ultragist_size, raw_size_to_cache, window_size)
        past_key_value = (key_states, value_states)

        key_position_ids = position_ids
        # align query position_ids with key
        query_position_ids = key_position_ids[:, -q_len:]

        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, query_position_ids)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # attn_output = self.o_proj_with_fastlora(attn_output, past_hidden_states=past_hidden_states)
        
        attn_output = self.fastlora_o_proj(attn_output, past_hidden_states=past_hidden_states)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MistralSdpaAttention(MistralAttention):
    """
    Mistral attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        past_hidden_states: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        bsz, q_len, _ = hidden_states.size()
        kv_seq_len = hidden_states.shape[-2]
        # past_key, past_value, ultragist_sizes, total_ultragist_size, raw_size_to_cache, window_size = past_key_value
        past_key, past_value = past_key_value

        if past_key is not None:
            past_seq_len = past_key.shape[2]
            kv_seq_len += past_seq_len
        else:
            past_seq_len = 0

        # query_states, key_states, value_states = self.qkv_proj_with_fastlora(hidden_states, past_hidden_states=past_hidden_states)

        query_states = self.fastlora_q_proj(hidden_states, past_hidden_states=past_hidden_states)
        key_states = self.fastlora_k_proj(hidden_states, past_hidden_states=past_hidden_states)
        value_states = self.fastlora_v_proj(hidden_states, past_hidden_states=past_hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # return keys and values before rope
        # NOTE: incrementally return keys and values for efficiency 
        # if window_size > 0:
        #     past_key_value = (key_states, value_states, ultragist_sizes, total_ultragist_size, raw_size_to_cache, window_size)

        if past_key is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        # NOTE: window_size == 0 indicates the ultragist is disabled, the model works as is, so the new past_key_values should concatenate old ones
        # if window_size == 0:
        #     past_key_value = (key_states, value_states, ultragist_sizes, total_ultragist_size, raw_size_to_cache, window_size)
        past_key_value = (key_states, value_states)

        key_position_ids = position_ids
        # align query position_ids with key
        query_position_ids = key_position_ids[:, -q_len:]

        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, query_position_ids)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        # attn_output = self.o_proj_with_fastlora(attn_output, past_hidden_states=past_hidden_states)

        attn_output = self.fastlora_o_proj(attn_output, past_hidden_states=past_hidden_states)

        return attn_output, None, past_key_value


MISTRAL_ATTENTION_CLASSES = {
    "eager": MistralAttention,
    "sdpa": MistralSdpaAttention,
}


class MistralDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MISTRAL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        past_hidden_states: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        # past_key, past_value, ultragist_sizes, total_ultragist_size, raw_size_to_cache, window_size = past_key_value
        # past_key, past_value, past_state, fastlora_rank = past_key_value

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            past_hidden_states=past_hidden_states,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(
            hidden_states,
            past_key_value=past_key_value,
            past_hidden_states=past_hidden_states,
        )
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


MISTRAL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MistralConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Mistral Model outputting raw hidden-states without any specific head on top.",
    MISTRAL_START_DOCSTRING,
)
class MistralPreTrainedModel(PreTrainedModel):
    config_class = MistralConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MistralDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        # if hasattr(module, "_is_ultragist_param") and module._is_ultragist_param:
        #     if torch.distributed.get_rank() == 0:
        #         print(module)
        #     module.weight.data.zero_()

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


MISTRAL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Mistral Model outputting raw hidden-states without any specific head on top.",
    MISTRAL_START_DOCSTRING,
)
class MistralModel(MistralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # ultragist: add ultragist embedding
        # self.ultragist_embed_tokens = nn.Embedding(1, config.hidden_size, self.padding_idx)
        # self.ultragist_embed_tokens._is_hf_initialized = True
        self.fastlora_embed_tokens = nn.Embedding(max(1, config.fastlora_gist_len), config.hidden_size)

        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _init_fastlora_embed(self, missing_keys):
        # randomize the fastlora embedding with default embedding initialization
        if is_deepspeed_zero3_enabled():
            import deepspeed
            with deepspeed.zero.GatheredParameters([self.fastlora_embed_tokens.weight], modifier_rank=0):
                self.fastlora_embed_tokens.weight.data.normal_(mean=0.0, std=1.0)
        else:
            if any("fastlora_embed_tokens" in missing_key for missing_key in missing_keys):
                self.fastlora_embed_tokens.weight.data.normal_(mean=0.0, std=1.0)

    # def _init_ultragist_embed(self, missing_keys):
    #     """Initialize the ultragist token embedding with that of the eos token."""
    #     if is_deepspeed_zero3_enabled():
    #         import deepspeed
    #         params = [self.ultragist_embed_tokens.weight, self.embed_tokens.weight]
    #         with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
    #             # deepspeed will initialize the parameters to zero
    #             if (self.ultragist_embed_tokens.weight == 0).all():
    #                 if self.config.ultragist_embed_init == "bos":
    #                     self.ultragist_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[self.config.bos_token_id]
    #                 elif self.config.ultragist_embed_init == "eos":
    #                     self.ultragist_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[self.config.eos_token_id]
    #                 else:
    #                     raise NotImplementedError(f"Make sure ultragist_embed_init is either eos or bos, found {self.config.ultragist_embed_init}")
    #     else:
    #         if any("ultragist_embed_tokens" in missing_key for missing_key in missing_keys):
    #             if self.config.ultragist_embed_init == "bos":
    #                 self.ultragist_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[self.config.bos_token_id]
    #             elif self.config.ultragist_embed_init == "eos":
    #                 self.ultragist_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[self.config.eos_token_id]
    #             else:
    #                 raise NotImplementedError(f"Make sure ultragist_embed_init is either eos or bos, found {self.config.ultragist_embed_init}")


    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        memory: Optional[Dict[str, torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # ultragist: always use cache
        # fastlora: always use cache
        # use_cache = True
        assert use_cache is True, "FastLora require cache to be enabled"
        assert output_hidden_states is True, "FastLora require output_hidden_states to be enabled"

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        # elif inputs_embeds is not None:
        #     batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # ultragist: create position_ids for all keys including past_keys
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if memory.get("past_hidden_states", None) is not None:
            past_keys, past_values = past_key_values[0]
            past_hidden_states = memory["past_hidden_states"]
            step_gist_len = memory["step_gist_len"]
            # past_keys, past_values, past_hidden_states, fastlora_rank = past_key_values[0]
            # print(f"past_keys: {past_keys.shape if past_keys is not None else None}, past_values: {past_values.shape if past_values is not None else None}, past_hidden_states: {past_hidden_states[0].shape if past_hidden_states[0] is not None else None}, step_gist_len: {step_gist_len}")
            past_key_values_length = past_keys.shape[2] if past_keys is not None else 0
            # print(f"past_key_values_length: {past_key_values_length}")
            seq_length_with_past = seq_length_with_past + past_key_values_length
            # print(f"seq_length_with_past: {seq_length_with_past}")
            ordinal_length = seq_length - step_gist_len
            # print(f"ordinal_length: {ordinal_length}")
        
            ordinal_input_ids = input_ids[:, :ordinal_length]
            # print(f"ordinal_input_ids: {ordinal_input_ids.shape}")
            special_input_inds = input_ids[:, ordinal_length:]
            # print(f"special_input_inds: {special_input_inds.shape}")
            ordinal_inputs_embeds = self.embed_tokens(ordinal_input_ids)
            # print(f"ordinal_inputs_embeds: {ordinal_inputs_embeds.shape}")
            fastlora_inputs_embeds = self.fastlora_embed_tokens(special_input_inds - self.config.vocab_size)
            # print(f"fastlora_inputs_embeds: {fastlora_inputs_embeds.shape}")
            inputs_embeds = torch.cat([ordinal_inputs_embeds, fastlora_inputs_embeds], dim=1)
            # print(f"inputs_embeds: {inputs_embeds.shape}")
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
            # print(f"attention_mask: {attention_mask.shape}")
            start_past, end_past = 0, past_key_values_length
            # print(f"start_past: {start_past}, end_past: {end_past}")
            start_ordinal, end_ordinal = end_past, past_key_values_length + ordinal_length
            # print(f"start_ordinal: {start_ordinal}, end_ordinal: {end_ordinal}")
            start_special, end_special = end_ordinal, seq_length_with_past
            # print(f"start_special: {start_special}, end_special: {end_special}")

            position_ids = torch.arange(seq_length_with_past, dtype=torch.long, device=device).repeat(batch_size, 1)
            position_ids[:, start_special:end_special] = torch.arange(step_gist_len, device=device)
            # print(f"position_ids: {position_ids.shape}")

            # min_value = torch.finfo(inputs_embeds.dtype).min
            # attention_mask[..., :seq_length, start_past:end_past] = min_value
            # print(f"attention_mask: {attention_mask.shape}")
        else:
            inputs_embeds = self.embed_tokens(input_ids)
            assert self._use_sdpa, "SDPA is required when ultragist is disabled"
            assert output_attentions == False, "output_attentions=True can not be supported when using SDPA"
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
            position_ids = torch.arange(seq_length_with_past, dtype=torch.long, device=device).repeat(batch_size, 1)

        
        # print(f"total_ultragist_size:  {total_ultragist_size}")
        # print(f"raw_size_to_cache:  {raw_size_to_cache}")
        # print(f"position_ids:       {position_ids}")
        # print(f"attention_mask:\n{attention_mask}")
        # x = input()
        # if x == "s":
        #     return

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        # ultragist: still use tuple to organize cache
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # ultragist: slice out the past_key_value of the corresponding layer
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            past_hidden_states = memory["past_hidden_states"][idx] if "past_hidden_states" in memory else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    past_hidden_states,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    past_hidden_states=past_hidden_states,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MistralForCausalLM(MistralPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Override the default from_pretrained to extend vocab size according to ultragist_size."""
        kwargs.update(output_loading_info=True)
        model, loading_info = super().from_pretrained(*args, **kwargs)

        # NOTE: set memory after from_pretrained because there may be another transformer model inside the Memory object, which may cause weird erros during loading
        # config = model.config
        # model.memory = Memory(
        #     model_config=config,
        #     k_seq_dim=2,
        #     v_seq_dim=2,
        # )

        missing_keys = loading_info["missing_keys"]
        # NOTE: the ultragist parameters may or may not be loaded from the checkpoint
        # if it is loaded from the checkpoint, we should not re-initilize it
        # model.model._init_ultragist_embed(missing_keys)
        # # initialize weights of possible q,k,v,o,mlp
        # for layer in model.model.layers:
        #     layer.self_attn._init_ultragist_proj(missing_keys)
        #     layer.mlp._init_ultragist_proj(missing_keys)
        model.model._init_fastlora_embed(missing_keys)
        for layer in model.model.layers:
            layer.self_attn._init_fastlora_proj(missing_keys, kwargs={"self_attn.v_proj": layer.self_attn.v_proj, "self_attn.k_proj": layer.self_attn.k_proj})
            layer.mlp._init_fastlora_proj(missing_keys, kwargs={"self_attn.v_proj": layer.self_attn.v_proj, "self_attn.k_proj": layer.self_attn.k_proj})

        return model

    def _native_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        memory: Optional[Dict[str, torch.Tensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        shift_labels: Optional[bool] = True,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # when we directly call _native_forward, the past_key_values would be None
        # if past_key_values is None:
        #     # NOTE: set window size to 0, so that new past_key_values are returned properly, see MistralAttention.forward
        #     past_key_values = [(None, None, [0], 0, 0, 0) for _ in range(self.config.num_hidden_layers)]
        if past_key_values is None:
            past_key_values = [(None, None) for _ in range(self.config.num_hidden_layers)]
        if memory is None:
            memory = {}

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            memory=memory,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        batch_loss = None
        valid_token_num = None
        
        if labels is not None:
            loss, batch_loss, valid_token_num = compute_loss(logits, labels, shift=shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ModelOutput(
            loss=loss,
            batch_loss=batch_loss,
            valid_token_num=valid_token_num,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _fastlora_forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
    ):

        # assert batch size is 1
        assert input_ids.shape[0] == 1
        start_idx = 0
        end_idx = 0
        total_input_length = input_ids.shape[1]
        # FIXME: change back to the full length
        # total_input_length = 2048
        # past_key_values: (key vectors, value vectors, lora vectors)
        if past_key_values is None:
            past_key_values = [(None, None) for _ in range(self.config.num_hidden_layers)]
        past_gist_key_values = [(None, None) for _ in range(self.config.num_hidden_layers)]
        past_hidden_states = [None for _ in range(self.config.num_hidden_layers)]
        vocab_size = self.config.vocab_size
        output_batch_loss = 0
        output_valid_token_num = 0
        output_logits = []
        step_idx = 0
        while end_idx < total_input_length:
            # FIXME: early stop the forward pass
            if step_idx >= 2:
                print("[WARNING] early stop the fastlora forward because of the step_idx >= 2")
                break
            # Update start_idx and end_idx
            start_idx = end_idx
            eos_token_id = self.config.eos_token_id
            while start_idx < input_ids.shape[1] and input_ids[0, start_idx] == eos_token_id:
                start_idx += 1
            end_idx = min(start_idx + self.config.fastlora_window, total_input_length)
            # NOTE: for pretraining, we assume there isn't any two consecutive eos token
            # if there are two eos token in a row, we should end the current segment at the first eos token
            is_eos = input_ids[0, start_idx:end_idx] == eos_token_id
            is_double_eos = is_eos[:-1] & is_eos[1:]
            if is_double_eos.any():
                end_idx = min(end_idx, start_idx + is_double_eos.nonzero()[0][0].item() + 1)

            step_input_ids = input_ids[:, start_idx: end_idx]
            step_attention_mask = attention_mask[:, start_idx: end_idx] if attention_mask is not None else torch.ones_like(step_input_ids)
            
            # NOTE: we do not compute the loss of the last token.
            step_labels = labels[:, start_idx: end_idx] if labels is not None else None
            
            # # NOTE: the first window is encoded without fastlora parameters, we should skip it when computing loss
            if self.training and step_idx <= 0 and step_labels is not None:
                step_labels[:] = -100
            
            step_gist_len = self.config.fastlora_gist_len
            # append special tokens to the end of the input_ids
            step_input_ids = torch.cat([
                step_input_ids, 
                torch.arange(vocab_size, vocab_size + step_gist_len, device=input_ids.device, dtype=input_ids.dtype).reshape(1, -1)
            ], dim=-1)
            step_attention_mask = torch.cat([step_attention_mask, step_attention_mask.new_ones(1, step_gist_len)], dim=-1)
            step_labels = torch.cat([step_labels, step_labels.new_full((1, step_gist_len), -100)], dim=-1) if step_labels is not None else None
            # append ones for past_key_values in attention_mask
            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values[0][0] is not None else 0
            if past_key_values_length > 0:
                step_attention_mask = torch.cat([step_attention_mask.new_ones(1, past_key_values_length), step_attention_mask], dim=-1)

            # print(f"step_input_ids: {step_input_ids.shape}, past_hidden_states: {past_hidden_states[0].shape if past_hidden_states[0] is not None else None}")

            # forward pass
            outputs = self._native_forward(
                input_ids=step_input_ids,
                attention_mask=step_attention_mask,
                position_ids=position_ids,
                # past_key_values=[(key, value, state, step_gist_len) for (key, value), state in zip(past_key_values, past_hidden_states)],
                past_key_values=past_key_values,
                memory={"step_gist_len": step_gist_len, "past_hidden_states": past_hidden_states, "past_gist_key_values": past_gist_key_values},
                inputs_embeds=inputs_embeds,
                labels=step_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            # from transformers import AutoTokenizer
            # if hasattr(self, "tokenizer") is False:
            #     self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
            # # decode the input_ids
            # logger.info(f"step_idx: {step_idx}, start_idx: {start_idx}, end_idx: {end_idx}, loss: {outputs.loss}, valid_token_num: {outputs.valid_token_num}")
            # # logger.info(f">>> INPUT >>> {self.tokenizer.decode(step_input_ids[0, :])} <<< INPUT <<<")
            # # step_labels_print = torch.where(step_labels == -100, torch.tensor(0, device=step_labels.device), step_labels)
            # # logger.info(f">>> LABEL >>> {self.tokenizer.decode(step_labels_print[0, :])} <<< LABEL <<<")

            # Update past_gist_key_values
            for i in range(len(past_key_values)):
                new_keys = outputs.past_key_values[i][0][:, :, -step_gist_len:]
                new_values = outputs.past_key_values[i][1][:, :, -step_gist_len:]
                past_key_values_layer_i = past_gist_key_values[i]
                if past_key_values_layer_i[0] is not None:
                    new_keys = torch.concat([past_key_values_layer_i[0], new_keys], dim=2)
                    new_values = torch.concat([past_key_values_layer_i[1], new_values], dim=2)
                past_gist_key_values[i] = (new_keys, new_values)
            past_key_values = [(None, None) for _ in range(self.config.num_hidden_layers)]

            # Update past_hidden_states
            for i in range(len(past_hidden_states)):
                new_states = outputs.hidden_states[i][:, -self.config.fastlora_attn_len:]
                past_hidden_states[i] = new_states

            # Update
            if outputs.loss is not None:
                output_batch_loss = output_batch_loss + outputs.loss * outputs.valid_token_num
                output_valid_token_num = output_valid_token_num + outputs.valid_token_num
            output_logits.append(outputs.logits[:, :end_idx - start_idx])
            assert outputs.logits.shape[1] == end_idx - start_idx + step_gist_len

            step_idx += 1
        
        if isinstance(output_valid_token_num, int):
            assert output_valid_token_num == 0
            loss, batch_loss = None, None
            output_valid_token_num = None
        else:
            loss = output_batch_loss.sum() / output_valid_token_num.sum()
            batch_loss = output_batch_loss / torch.maximum(output_valid_token_num, torch.tensor(1, device=output_valid_token_num.device))
        # print("loss", output_batch_loss / output_valid_token_num, "output_valid_token_num", output_valid_token_num, "labels non zero entries", (labels != -100).sum(), )

        return ModelOutput(
            loss=loss, batch_loss=batch_loss,
            valid_token_num=output_valid_token_num,
            logits=torch.cat(output_logits, dim=1),
        )


    # def _ultragist_forward(self, 
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    # ):
    #     # initialize cache
    #     self.memory.prepare(
    #         input_ids=input_ids, 
    #         attention_mask=attention_mask, 
    #         labels=labels
    #     )

    #     while not self.memory.finish:
    #         input_ids, attention_mask, past_key_values, labels = self.memory.step()

    #         # NOTE: the first window is encoded without ultragist parameters, we should skip it when computing loss
    #         if self.training and self.memory._step_idx == 1:
    #             labels[:] = -100

    #         outputs = self._native_forward(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             past_key_values=past_key_values,
    #             inputs_embeds=inputs_embeds,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=True | output_hidden_states,
    #             return_dict=return_dict,
    #             labels=labels,
    #             # NOTE: the labels have been shifted so that all tokens in the window have the proper loss
    #             shift_labels=False,
    #         )

    #         # update past_key_values
    #         self.memory.update_memory(outputs.past_key_values)

    #         if labels is not None:
    #             # if torch.distributed.get_rank() == 0:
    #             #     print(outputs.batch_loss, outputs.valid_token_num)
    #             # update loss and past_key_values
    #             self.memory.update_loss(outputs.batch_loss, outputs.valid_token_num)

    #     # output loss, past_key_values, and perplexity
    #     outputs = self.memory.output(outputs)
    #     return outputs
    
    def forward(self, **kwargs):
        """Forward computation over a batch of sequences.
        """
        # only allow gradient when training
        with optional_grad_ctx(with_grad=self.training):
            # we can disable ultragist to use the original mistral
            # if hasattr(self, "_enable_ultragist") and self._enable_ultragist == False:
            #     return self._native_forward(**kwargs)
            # else:
            #     return self._ultragist_forward(**kwargs)
            return self._fastlora_forward(**kwargs)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        raise NotImplementedError("Generation is not supported")
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
