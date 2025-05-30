import math
import torch
from tqdm import tqdm
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Mapping, Optional, Tuple
from accelerate import Accelerator
from collections import defaultdict
from transformers.modeling_outputs import BaseModelOutputWithPast
from datasets import load_dataset


def optional_grad_ctx(with_grad=False):
    if with_grad:
        return nullcontext()
    else:
        return torch.no_grad()

def move_to_device(data, device):
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: move_to_device(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(move_to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        return data.to(**kwargs)
    else:
        return data

def compute_loss(logits, labels, shift=False):
    """
    Returns:
        token_loss: batch_size, seq_length
    """
    if shift:
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

    labels = labels.to(logits.device)
    batch_size = logits.shape[0]

    # NOTE: the loss on -100 labels is 0 by default
    token_loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), 
        labels.reshape(-1), 
        reduction="none"
    ).reshape(batch_size, -1)   # batch_size, seq_len
    
    valid_token_num = (labels != -100).sum(-1)  # batch_size
    all_valid_token_num = valid_token_num.sum()
    
    if all_valid_token_num > 0:
        loss = token_loss.sum() / valid_token_num.sum()
    else:
        loss = token_loss.sum()

    batch_loss = token_loss.sum(-1) / valid_token_num
    # prevent nan
    if (valid_token_num == 0).any():
        batch_loss = batch_loss.masked_fill(valid_token_num == 0, 0.)

    return loss, batch_loss, valid_token_num


def compute_loss_context_input(model, input_ids_seq1, input_ids_seq2, attention_mask_seq1, attention_mask_seq2, label_seq1, label_seq2):
    from torch.nn.utils.rnn import pad_sequence
    import torch.nn.functional as F

    # cut the context into segments
    window_size = model.peft_config['default'].fastlora_window
    pad_token_id = model.config.pad_token_id if model.config.pad_token_id else model.config.eos_token_id
    
    number_windows = (input_ids_seq1.shape[-1] + window_size - 1) // window_size
    seq_len = (input_ids_seq1.shape[-1] + number_windows - 1) // number_windows
    input_ids_seq1 = F.pad(input_ids_seq1, (0, number_windows * seq_len - input_ids_seq1.shape[-1]), value=pad_token_id).reshape(-1, number_windows, seq_len)
    attention_mask_seq1 = F.pad(attention_mask_seq1, (0, number_windows * seq_len - attention_mask_seq1.shape[-1]), value=0).reshape(-1, number_windows, seq_len)
    label_seq1 = F.pad(label_seq1, (0, number_windows * seq_len - label_seq1.shape[-1]), value=-100).reshape(-1, number_windows, seq_len)
    
    # print("model.device", model.device)
    # print(f'{input_ids_seq1.shape} ({input_ids_seq1.dtype}, {input_ids_seq1.device}), {attention_mask_seq1.shape} ({attention_mask_seq1.dtype}, {attention_mask_seq1.device}), {label_seq1.shape} ({label_seq1.dtype}, {label_seq1.device})')
    # print(f'{input_ids_seq2.shape} ({input_ids_seq2.dtype}, {input_ids_seq2.device}), {attention_mask_seq2.shape} ({attention_mask_seq2.dtype}, {attention_mask_seq2.device}), {label_seq2.shape} ({label_seq2.dtype}, {label_seq2.device})')

    # >>> Mode 1: default
    assert input_ids_seq1.shape[0] == 1, "batch size should be 1"
    input_ids = pad_sequence([*input_ids_seq1.squeeze(0), input_ids_seq2.squeeze(0)], batch_first=True, padding_value=pad_token_id).unsqueeze(0)
    attention_mask = pad_sequence([*attention_mask_seq1.squeeze(0), attention_mask_seq2.squeeze(0)], batch_first=True, padding_value=0).unsqueeze(0)
    labels = pad_sequence([*label_seq1.squeeze(0), label_seq2.squeeze(0)], batch_first=True, padding_value=-100).unsqueeze(0)
    inputs = {}
    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention_mask
    inputs["labels"] = labels

    # inputs["labels"] = inputs["input_ids"]
    # print(inputs["input_ids"].shape, inputs["input_ids"])
    # print(inputs["attention_mask"].shape, inputs["attention_mask"])
    # print(inputs["labels"].shape, inputs["labels"])
    # print(inputs)
    # for name, x in model.named_parameters():
    #     print(f"{name: ^80} {x.dtype}, {x.device}")

    outputs = model(**inputs)

    logits = outputs.logits
    loss, batch_loss, valid_token_num = compute_loss(logits.reshape((-1,) + logits.shape[-2:]), labels.reshape(-1, labels.shape[-1]), shift=True)
    batch_loss = batch_loss * valid_token_num
    batch_loss = batch_loss.reshape(logits.shape[0], logits.shape[1])
    valid_token_num = valid_token_num.reshape(logits.shape[0], logits.shape[1])
    valid_token_num = valid_token_num.sum(-1)
    batch_loss = batch_loss.sum(-1) / torch.clamp(valid_token_num, min=1)
    return loss, batch_loss, valid_token_num


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, reconstruct_tokens=None, context_len=1024, input_len=1024):
    
    data = load_dataset("json", data_files="../../data/pretrain/val-8K-1M/data.json")
    data = data["train"]

    # if accelerator is not None and type(dataloader) == torch.utils.data.DataLoader:
    #     # if the dataloader has been prepared, we shall not prepare it twice, especially in case of deepspeed
    #     dataloader = accelerator.prepare(dataloader)

    # if accelerator.process_index == 0:
    #     for name, x in model.named_parameters():
    #         print(f"{name: ^80} {x.dtype}")

    # all_loss = defaultdict(list)
    all_loss = []
    for i, x in enumerate(tqdm(data, desc="Computing Perplexity")):
        x = tokenizer(x["text"], return_tensors="pt")

        # prepare the context and the input
        input_ids_seq_1, input_ids_seq_2 = x["input_ids"][:, :context_len], x["input_ids"][:, context_len:]
        attention_mask_seq_1, attention_mask_seq_2 = x["attention_mask"][:, :context_len], x["attention_mask"][:, context_len:]
        label_seq_1, label_seq_2 = x["input_ids"][:, :context_len], x["input_ids"][:, context_len:]
        input_ids_seq_2 = input_ids_seq_2[:, :input_len]
        attention_mask_seq_2 = attention_mask_seq_2[:, :input_len]
        label_seq_2 = label_seq_2[:, :input_len]
        
        if reconstruct_tokens is not None:
            input_ids_seq_2, attention_mask_seq_2, label_seq_2 = input_ids_seq_1, attention_mask_seq_1, label_seq_1     # for reconstruction evaluation
            input_ids_seq_2, attention_mask_seq_2, label_seq_2 = input_ids_seq_2[:, :reconstruct_tokens], attention_mask_seq_2[:, :reconstruct_tokens], label_seq_2[:, :reconstruct_tokens]     # for short instruction evaluation
        # input_ids_seq_1, attention_mask_seq_1 = input_ids_seq_1[:, :64], attention_mask_seq_1[:, :64]     # for short context evaluation
        label_seq_1 = torch.full_like(input_ids_seq_1, -100)
        
        input_ids_seq_1 = input_ids_seq_1.to(model.device)
        input_ids_seq_2 = input_ids_seq_2.to(model.device)
        attention_mask_seq_1 = attention_mask_seq_1.to(model.device)
        attention_mask_seq_2 = attention_mask_seq_2.to(model.device)
        label_seq_1 = label_seq_1.to(model.device)
        label_seq_2 = label_seq_2.to(model.device)

        # print(input_ids_seq_1.shape, input_ids_seq_2.shape, attention_mask_seq_1.shape, attention_mask_seq_2.shape, label_seq_1.shape, label_seq_2.shape)

        loss, batch_loss, valid_token_num = compute_loss_context_input(model, input_ids_seq_1, input_ids_seq_2, attention_mask_seq_1, attention_mask_seq_2, label_seq_1, label_seq_2)

        all_loss.append(loss.item())
    
    perplexity = math.exp(sum(all_loss) / len(all_loss))
    return perplexity

@torch.no_grad()
def evaluate_squad(model, tokenizer):
    data = load_dataset("json", data_files="../../data/pretrain/eval-squad-100/data_eval_squad.jsonl")
    data = data["train"]

    all_loss = []
    # item = data[0]
    for item in tqdm(data):
        context_text = f"Title: {item['title']}\nPassage: {item['context']}"
        input_text = item['question']
        answer_text = item["answers"]["text"][0]

        context_text_ids = tokenizer(context_text, return_tensors="pt").input_ids
        input_text_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
        input_answer_text_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}, {"role": "assistant", "content": answer_text}],
            tokenize=True, return_tensors="pt",
        )
        assert (input_answer_text_ids[:, :input_text_ids.shape[1]] == input_text_ids).all()

        num_label_tokens = input_answer_text_ids.shape[1] - input_text_ids.shape[1]

        context_text_ids = context_text_ids.to(model.device)
        input_answer_text_ids = input_answer_text_ids.to(model.device)
        input_ids_seq_1 = context_text_ids
        input_ids_seq_2 = input_answer_text_ids
        attention_mask_seq_1 = torch.ones_like(input_ids_seq_1)
        attention_mask_seq_2 = torch.ones_like(input_ids_seq_2)
        label_seq_1 = torch.full_like(input_ids_seq_1, -100)
        label_seq_2 = input_ids_seq_2.clone()
        label_seq_2[:, :-num_label_tokens] = -100

        loss, batch_loss, valid_token_num = compute_loss_context_input(model, input_ids_seq_1, input_ids_seq_2, attention_mask_seq_1, attention_mask_seq_2, label_seq_1, label_seq_2)

        all_loss.append(loss.item())

    return sum(all_loss) / len(all_loss)

@torch.no_grad()
def evaluate_generation(model, dataloader, accelerator:Optional[Accelerator]=None, tokenizer=None, return_new_tokens_only=True, return_decoded=True, **generation_config):
    if accelerator is not None and type(dataloader) == torch.utils.data.DataLoader:
        # if the dataloader has been prepared, we shall not prepare it twice, especially in case of deepspeed
        dataloader = accelerator.prepare(dataloader)

    all_indices = []
    all_outputs = []
    
    for i, x in enumerate(tqdm(dataloader, desc="Computing Generation")):
        # if i > 3:
        #     break
        
        # NOTE: important to reset memory for every batch
        if hasattr(model, "memory"):
            model.memory.reset()

        indices = x.pop("index")
        # length is used to group training data, no use here
        length = x.pop("length", None)

        outputs = model.generate(**x, **generation_config)
        if return_new_tokens_only:
            start_idx = x["input_ids"].shape[1]
            outputs = outputs[:, start_idx:]

        if accelerator is not None and accelerator.num_processes > 1:
            # must be contiguous
            outputs = accelerator.pad_across_processes(outputs.contiguous(), pad_index=tokenizer.pad_token_id, dim=1)
            outputs = accelerator.gather_for_metrics(outputs)
            indices = accelerator.gather_for_metrics(indices)

        outputs = outputs.tolist()
        indices = indices.tolist()
        if return_decoded:
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_indices.extend(indices)
        all_outputs.extend(outputs)

    return all_indices, all_outputs


@torch.no_grad()
def evaluate_nll(model, dataloader, accelerator:Optional[Accelerator]=None):
    if accelerator is not None and type(dataloader) == torch.utils.data.DataLoader:
        # if the dataloader has been prepared, we shall not prepare it twice, especially in case of deepspeed
        dataloader = accelerator.prepare(dataloader)

    # if accelerator.process_index == 0:
    #     for name, x in model.named_parameters():
    #         print(f"{name: ^80} {x.dtype}")

    all_loss = defaultdict(list)
    for i, x in enumerate(tqdm(dataloader, desc="Computing Perplexity")):
        # NOTE: important to reset memory for every batch
        if hasattr(model, "memory"):
            model.memory.reset()

        # the seq id
        index = x.pop("index")
        # length is used to group training data, no use here
        length = x.pop("length", None)

        output = model(**x)

        # NOTE: we need the loss for each element in the batch for accurate computation, because the number of valid tokens may differ among elements
        if hasattr(output, "batch_loss"):
            # output from our model has batch_loss by default
            batch_loss = output.batch_loss
            valid_token_num = output.valid_token_num
        else:
            # output from other models does not
            loss, batch_loss, valid_token_num = compute_loss(output.logits, x["labels"], shift=True)

        if accelerator is not None and accelerator.num_processes > 1:
            # num_device * batch_size
            index = accelerator.gather_for_metrics(index)
            batch_loss = accelerator.gather_for_metrics(batch_loss)
            valid_token_num = accelerator.gather_for_metrics(valid_token_num)

        for _id, _loss in zip(index.tolist(), batch_loss.tolist()):
            # loss times num is the total loss of all valid tokens
            all_loss[_id].append(_loss)

    return all_loss



@dataclass
class ModelOutput(BaseModelOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    batch_loss: Optional[torch.FloatTensor] = None
    valid_token_num: Optional[torch.LongTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
