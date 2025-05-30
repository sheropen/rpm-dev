import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
import string
from tqdm import tqdm

def normalize_text(text, ignore_case=True, ignore_punctuation=True, ignore_space=True, ignore_number=False):
    if isinstance(text, str):
        text = [text]
        unpack = True
    else:
        unpack = False
    if ignore_case:
        text = np.char.lower(text)
    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        text = np.char.translate(text, table=repl_table)
    if ignore_number:
        repl_table = string.digits.maketrans("", "", string.digits)
        text = np.char.translate(text, table=repl_table)
    if ignore_space:
        for i, words in enumerate(np.char.split(text)):
            text[i] = " ".join(words)
    if isinstance(text, np.ndarray):
        text = text.tolist()
    if unpack:
        text = text[0]
    return text 

def estimate_flops(context_len, input_len, dim=4096, n_layers=32, head_dim=128, hidden_dim=14336, n_heads=32, n_kv_heads=8, window_size=4096, vocab_size=32000):
    # dim = 4096
    # n_layers = 32
    # head_dim = 128
    # hidden_dim = 14336
    # n_heads = 32
    # n_kv_heads = 8
    # window_size = 4096
    # vocab_size = 32000

    op_embed = input_len * dim
    op_qkv = input_len * dim * (head_dim * n_heads + head_dim * n_kv_heads + head_dim * n_kv_heads) * n_layers
    op_mask = input_len * (input_len * context_len) * dim * n_layers
    op_proj = input_len * dim * dim * n_layers
    op_ffn = input_len * dim * hidden_dim * 3 * n_layers
    op_deembed = input_len * dim * vocab_size

    op_total = op_embed + op_qkv + op_mask + op_proj + op_ffn + op_deembed

    # print("context_len:", context_len)
    # print("input_len:", input_len)
    # print("op_embed:", op_embed / 1e12)
    # print("op_qkv:", op_qkv / 1e12)
    # print("op_mask:", op_mask / 1e12)
    # print("op_proj:", op_proj / 1e12)
    # print("op_ffn:", op_ffn / 1e12)
    # print("op_deembed:", op_deembed / 1e12)
    # print("TFLOPs:", op_total / 1e12)

    return op_total / 1e12

def load_model_and_tokenizer(model_name, device='cpu', fastlora_merge="pre-norm-sum", **kwargs):
    if "fastlora" in model_name:
        from fastlora.config import FastLoraConfig
        from fastlora.model import FastLoraModelForCausalLM, FastLoraModel, get_peft_model_state_dict, set_peft_model_state_dict, load_pretrained_model
        import peft.peft_model as peft_model
        import peft.mapping as peft_mapping
        ## monkey patching
        peft_model.PEFT_TYPE_TO_MODEL_MAPPING.update({"FASTLORA": FastLoraModel})
        peft_mapping.PEFT_TYPE_TO_CONFIG_MAPPING.update({"FASTLORA": FastLoraConfig})
        peft_model.get_peft_model_state_dict = get_peft_model_state_dict
        peft_model.set_peft_model_state_dict = set_peft_model_state_dict
        model = load_pretrained_model(
            model_name_or_path=model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            fastlora_params={"fastlora_merge": fastlora_merge},
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif "ultragist" in model_name:
        print("Loading UltraGist model")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="sdpa",
            # load the entire model on the default gpu
            device_map={"": "cuda"}, 
            # you can manually set the compression ratio, otherwise the model will automatically choose the most suitable compression ratio from [2,4,8,16,32]
            ultragist_ratio=[2],
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
        )
        # from ultragist.llama import LlamaForCausalLM
        # model = LlamaForCausalLM.from_pretrained(
        #     model_name, 
        #     #   trust_remote_code=True, 
        #     torch_dtype=torch.bfloat16, 
        #     attn_implementation="sdpa",
        #     # load the entire model on the default gpu
        #     device_map={"": "cuda"}, 
        #     # you can manually set the compression ratio, otherwise the model will automatically choose the most suitable compression ratio from [2,4,8,16,32]
        #     ultragist_ratio=[kwargs["ultragist_ratio"]] if "ultragist_ratio" in kwargs else [],
        # ).eval()
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            # attn_implementation="flash_attention_2",
        ).to(device).eval()
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            # FIXME: debug only
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
            print(f'[WARNING] Using tokenizer from "mistralai/Mistral-7B-Instruct-v0.2"')
    return model, tokenizer

@torch.inference_mode()
def ultragist_generate(args, model, tokenizer, context_text, input_text, device='cpu', max_new_tokens=20, stop=["\n"]):
    
    if args.get("insert_paddings", False):
        # append whitespace if the context_text is not empty

        num_whitespaces = model.config.ultragist_window - len(tokenizer.encode(context_text)) % model.config.ultragist_window + 10
        # print(num_whitespaces)
        context_text = context_text + "\n" * num_whitespaces

    input_text = context_text + "\n\n" + input_text
    if args.get("use_chat", False):
        input_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=False,
        )
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)

    # print(tokenizer.decode(input_ids[0]))

    # reset memory before new compression task
    model.memory.reset()

    # directly call generate to progressively compress the context while generating next tokens
    outputs = model.generate(
        input_ids,
        max_length=len(input_ids[0]) + max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )
    new_tokens = outputs.sequences[0][len(input_ids[0]):]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # find the first stop token in the output_text and return the text before it
    for s in stop:
        if s in output_text:
            output_text = output_text.split(s)[0]
            break
    
    # if args.get("insert_paddings", False):
    #     # extract the compressed memory (including the generated tokens)
    #     compressed_memory = model.memory.get_memory()
    ultragist_size, raw_size, sink_size = model.memory.get_memory_size()
    metainfo = {
        "ultragist_size": ultragist_size,
        "raw_size": raw_size,
        "sink_size": sink_size,
    }
    #     print(f"UltraGist size:   {ultragist_size}")
    #     print(f"Raw size:         {raw_size}")
    #     print(f"Sink size:        {sink_size}")
    #     print(f"Memory:           {compressed_memory[0][0].shape}")
    #     print("*"*20)

    return output_text.strip(), metainfo

@torch.inference_mode()
def default_generate(args, model, tokenizer, context_text, input_text, device='cpu', max_new_tokens=20, stop=["\n"]):
    input_text = context_text + "\n\n" + input_text
    if args.get("use_chat", False):
        input_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=False,
        )
    # print(input_text)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)

    outputs = model.generate(
        input_ids, 
        max_length=len(input_ids[0]) + max_new_tokens, 
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True
    )
    # print(tokenizer.decode(outputs.sequences[0]))
    new_tokens = outputs.sequences[0][len(input_ids[0]):]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # find the first stop token in the output_text and return the text before it
    for s in stop:
        if s in output_text:
            output_text = output_text.split(s)[0]
            break
    context_ids = tokenizer(context_text, return_tensors="pt").input_ids
    flops = estimate_flops(len(context_ids[0]), len(outputs.sequences[0]) - len(context_ids[0]))
    return output_text.strip(),  {"flops": flops}

@torch.inference_mode()
def fastlora_generate(model, tokenizer, context_text, input_text, device='cpu', use_chat=False, max_new_tokens=20, stop=["\n"], mode='weights'):
    context_input_ids = tokenizer(context_text, return_tensors="pt").input_ids
    context_input_ids = context_input_ids.to(device)
    if use_chat:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    
    input_ids_seq_1 = context_input_ids.unsqueeze(1)
    attention_mask_seq_1 = torch.ones_like(input_ids_seq_1)
    input_ids_seq_2 = input_ids
    # >>> Generation >>>
    outputs = model(
        input_ids=input_ids_seq_1, attention_mask=attention_mask_seq_1,
        output_hidden_states=True,
    )
    hidden_states_seq_1 = outputs.hidden_states
    if mode == 'weights':
        lora_weights = model.to_lora_weights(hidden_states_seq_1, attention_mask_seq_1)
        outputs = model.generate(
            inputs=input_ids_seq_2,
            fastlora_weights=lora_weights, 
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )
    elif mode == 'states':
        outputs = model.generate(
            inputs=input_ids_seq_2,
            fastlora_hidden_states_and_mask=(hidden_states_seq_1, attention_mask_seq_1),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )
    new_tokens = outputs.sequences[0][len(input_ids[0]):]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # find the first stop token in the output_text and return the text before it
    for s in stop:
        if s in output_text:
            output_text = output_text.split(s)[0]
            break
    # flops = estimate_flops(len(context_input_ids[0]), len(outputs.sequences[0]))
    # print(flops)
    return output_text.strip()

@torch.inference_mode()
def fastlora_perplexity(model, tokenizer, context_text, input_text, device='cpu', mode='weights'):

    inputs_seq_1 = tokenizer(context_text, return_tensors="pt")
    input_ids_seq_1, attention_mask_seq_1 = inputs_seq_1.input_ids.to(device), inputs_seq_1.attention_mask.to(device)
    inputs_seq_2 = tokenizer(input_text, return_tensors="pt")
    input_ids_seq_2, attention_mask_seq_2 = inputs_seq_2.input_ids.to(device), inputs_seq_2.attention_mask.to(device)

    if mode == 'default':
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
        print(outputs.logits[0, 0, :8, :8])
        print(outputs.logits[0, 1, :8, :8])
        logits = outputs.logits
        loss, batch_loss, valid_token_num = compute_loss(logits.reshape((-1,) + logits.shape[-2:]), labels.reshape(-1, labels.shape[-1]), shift=True)
        print(loss, batch_loss, valid_token_num)
        # all_loss.extend(batch_loss.tolist())
        all_loss.append(loss.item())
        # <<< Mode 1 <<<
    elif mode == 'weights':
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
    elif mode == "weights":
        # >>> Mode 3: weights >>>
        outputs = model(
            input_ids=input_ids_seq_1, attention_mask=attention_mask_seq_1,
            output_hidden_states=True,
        )
        hidden_states_seq_1 = outputs.hidden_states
        lora_weights = model.to_lora_weights(hidden_states_seq_1, attention_mask_seq_1)
        # print(hidden_states_seq_1[0].shape)
        # print(attention_mask_seq_1.shape)
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

@torch.inference_mode()
def fastlora_generate_adaptor(model, tokenizer, context_text, merge_strategy, max_window_size=None):

    device = model.device

    context_inputs = tokenizer(context_text, return_tensors='pt')
    context_input_ids = context_inputs.input_ids
    context_attention_mask = context_inputs.attention_mask

    ## Step 1:
    if merge_strategy == "concat":
        context_input_ids = context_input_ids.reshape(1, 1, -1).to(device)
        context_attention_mask = context_attention_mask.reshape(1, 1, -1).to(device)
        print(f'shape of context_input_ids: {context_input_ids.shape}')
        print(f'shape of context_attention_mask: {context_attention_mask.shape}')
        outputs = model(
            input_ids=context_input_ids, attention_mask=context_attention_mask,
            output_hidden_states=True,
        )
        context_hidden_states = outputs.hidden_states
        lora_weights = model.to_lora_weights(context_hidden_states, context_attention_mask)
    elif merge_strategy == "parallel" or merge_strategy == "sequential":
        assert max_window_size is not None
        num_chunk = (context_input_ids.shape[-1] + max_window_size - 1) // max_window_size
        window_size = (context_input_ids.shape[-1] + num_chunk - 1) // num_chunk
        num_pad = num_chunk * window_size - context_input_ids.shape[-1]
        context_input_ids = torch.nn.functional.pad(context_input_ids, (0, num_pad), value=tokenizer.pad_token_id)
        context_attention_mask = torch.nn.functional.pad(context_attention_mask, (0, num_pad), value=0)
        if merge_strategy == "parallel":
            context_input_ids = context_input_ids.reshape(num_chunk, 1, -1).to(device)
            context_attention_mask = context_attention_mask.reshape(num_chunk, 1, -1).to(device)
            print(f'shape of context_input_ids: {context_input_ids.shape}')
            print(f'shape of context_attention_mask: {context_attention_mask.shape}')
            outputs = model(
                input_ids=context_input_ids,
                attention_mask=context_attention_mask,
                output_hidden_states=True,
            )
            context_hidden_states = list(outputs.hidden_states)
            for i in range(len(context_hidden_states)):
                context_hidden_states[i] = context_hidden_states[i].transpose(0, 1)
            context_hidden_states = tuple(context_hidden_states)
            context_attention_mask = context_attention_mask.transpose(0, 1)
        elif merge_strategy == "sequential":
            context_input_ids = context_input_ids.reshape(1, num_chunk, -1).to(device)
            context_attention_mask = context_attention_mask.reshape(1, num_chunk, -1).to(device)
            print(f'shape of context_input_ids: {context_input_ids.shape}')
            print(f'shape of context_attention_mask: {context_attention_mask.shape}')
            outputs = model(
                input_ids=context_input_ids,
                attention_mask=context_attention_mask,
                output_hidden_states=True,
            )
            context_hidden_states = outputs.hidden_states
        lora_weights = model.to_lora_weights(context_hidden_states, context_attention_mask)
    elif merge_strategy == "sequential-long":
        assert max_window_size is not None

        num_chunk = (context_input_ids.shape[-1] + max_window_size - 1) // max_window_size
        window_size = (context_input_ids.shape[-1] + num_chunk - 1) // num_chunk
        num_pad = num_chunk * window_size - context_input_ids.shape[-1]
        context_input_ids = torch.nn.functional.pad(context_input_ids, (0, num_pad), value=tokenizer.pad_token_id)
        context_attention_mask = torch.nn.functional.pad(context_attention_mask, (0, num_pad), value=0)

        context_input_ids = context_input_ids.reshape(num_chunk, 1, -1).to(device)
        context_attention_mask = context_attention_mask.reshape(num_chunk, 1, -1).to(device)

        outer_product_dict = None
        lora_weights = None
        for i in tqdm(range(num_chunk), desc="Sequential Long"):
            outputs = model(
                input_ids=context_input_ids[i],
                attention_mask=context_attention_mask[i],
                output_hidden_states=True,
                fastlora_weights=lora_weights,
            )
            context_hidden_states = [x.unsqueeze(1) for x in outputs.hidden_states]
            lora_weights = model.to_lora_weights(
                context_hidden_states,
                context_attention_mask[i].unsqueeze(1),
                outer_product=outer_product_dict,
                return_outer_product=True,
            )
            outer_product_dict = {k: v["outer_product"] for k, v in lora_weights.items()}
            for k, v in lora_weights.items():
                v.pop("outer_product")
        
    else:
        raise ValueError(f"Invalid merge strategy: {merge_strategy}")

        # hidden_states_list = []
        # for i in range(len(context_list)):
        #     outputs = model(
        #         input_ids=context_input_ids[i].reshape(1, 1, -1), attention_mask=context_attention_mask[i].reshape(1, 1, -1),
        #         output_hidden_states=True,
        #     )
        #     hidden_states_list.append(outputs.hidden_states)
        # context_hidden_states = torch.stack(hidden_states_list, dim=1)
        # context_attention_mask = context_attention_mask.unsqueeze(0)
    
    # if mode == 'weights':
    #     lora_weights = model.to_lora_weights(context_hidden_states, context_attention_mask)
    # else:
    #     lora_weights = None


    return lora_weights

@torch.inference_mode()
def fastlora_conditional_generate(model, tokenizer, input_text=None, input_ids=None, use_chat=True, mode="weights", lora_weights=None, context_hidden_states=None, context_attention_mask=None, return_input_text=False, return_token_probs=False, stop=None, **kwargs):
    
    device = model.device

    if input_text is not None:
        if use_chat:
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": input_text}],
                tokenize=True, add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        assert isinstance(input_ids, torch.Tensor)
        input_ids = input_ids.to(device)
    else:
        assert input_ids is not None

    if mode == 'weights':
        assert lora_weights is not None
        outputs = model.generate(
            inputs=input_ids,
            fastlora_weights=lora_weights, 
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs
        )
    elif mode == 'states':
        assert context_hidden_states is not None
        assert context_attention_mask is not None
        outputs = model.generate(
            inputs=input_ids,
            fastlora_hidden_states_and_mask=(context_hidden_states, context_attention_mask),
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    new_tokens = outputs.sequences[0][len(input_ids[0]):]

    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # find the first stop token in the output_text and return the text before it
    if stop is not None:
        for s in stop:
            if s in output_text:
                output_text = output_text.split(s)[0]
    
    # output_text = output_text.strip()

    # if return_input_text:
    #     return output_text, tokenizer.decode(input_ids[0], skip_special_tokens=False)
    # else:
    #     return output_text

    return_tuple = ()
    if return_token_probs:
        token_probs = torch.nn.functional.softmax(outputs.scores[0], dim=-1)
        return_tuple += (token_probs,)
    else:
        return_tuple += (output_text.strip(),)

    if return_input_text:
        return_tuple += (tokenizer.decode(input_ids[0], skip_special_tokens=False),)
    
    return return_tuple if len(return_tuple) > 1 else return_tuple[0]

@torch.inference_mode()
def default_conditional_generate(model, tokenizer, input_text=None, context_text=None, use_chat=True, return_input_text=False, return_token_probs=False, stop=["\n"], **kwargs):
    device = model.device

    if context_text is not None:
        input_text = context_text + "\n\n" + input_text

    if use_chat:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    assert isinstance(input_ids, torch.Tensor)
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
        **kwargs
    )
    # print(tokenizer.decode(outputs.sequences[0]))
    new_tokens = outputs.sequences[0][len(input_ids[0]):]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # find the first stop token in the output_text and return the text before it
    for s in stop:
        if s in output_text:
            output_text = output_text.split(s)[0]
    # context_ids = tokenizer(context_text, return_tensors="pt").input_ids
    # flops = estimate_flops(len(context_ids[0]), len(outputs.sequences[0]) - len(context_ids[0]))
    # return output_text.strip(),  {"flops": flops}


    return_tuple = ()
    if return_token_probs:
        token_probs = torch.nn.functional.softmax(outputs.scores[0], dim=-1)
        return_tuple += (token_probs,)
    else:
        return_tuple += (output_text.strip(),)

    if return_input_text:
        return_tuple += (tokenizer.decode(input_ids[0], skip_special_tokens=False),)
    
    return return_tuple if len(return_tuple) > 1 else return_tuple[0]


@torch.inference_mode()
def ultragist_conditional_generate(model, tokenizer, context_text=None, input_text=None, use_chat=True, return_input_text=False, max_new_tokens=20, stop=["\n"], **kwargs):
    device = model.device
    
    num_whitespaces = model.config.ultragist_window - len(tokenizer.encode(context_text)) % model.config.ultragist_window + 10
    context_text = context_text + "\n" * num_whitespaces
    input_text = context_text + "\n\n" + input_text

    if use_chat:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    assert isinstance(input_ids, torch.Tensor)
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    # print(tokenizer.decode(input_ids[0]))

    # reset memory before new compression task
    model.memory.reset()

    # directly call generate to progressively compress the context while generating next tokens
    outputs = model.generate(
        input_ids,
        max_length=len(input_ids[0]) + max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        **kwargs
    )
    new_tokens = outputs.sequences[0][len(input_ids[0]):]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # find the first stop token in the output_text and return the text before it
    for s in stop:
        if s in output_text:
            output_text = output_text.split(s)[0]
            break
    
    # if args.get("insert_paddings", False):
    #     # extract the compressed memory (including the generated tokens)
    #     compressed_memory = model.memory.get_memory()
    ultragist_size, raw_size, sink_size = model.memory.get_memory_size()
    metainfo = {
        "ultragist_size": ultragist_size,
        "raw_size": raw_size,
        "sink_size": sink_size,
    }
    #     print(f"UltraGist size:   {ultragist_size}")
    #     print(f"Raw size:         {raw_size}")
    #     print(f"Sink size:        {sink_size}")
    #     print(f"Memory:           {compressed_memory[0][0].shape}")
    #     print("*"*20)
    if return_input_text:
        return output_text.strip(), tokenizer.decode(input_ids[0], skip_special_tokens=False), metainfo
    else:
        return output_text.strip()