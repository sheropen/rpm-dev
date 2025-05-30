import re
import os
import json
import math
import random
import datasets
from tqdm import tqdm
from functools import partial
from glob import glob
from contextlib import nullcontext
from transformers.utils import logging
from .utils import add_eos
from .chat import apply_chat_template
from .utils import split_into_chunks, merge_chunks
import torch
import torch.nn.functional as F

logger = logging.get_logger(__name__)


# RETRIEVAL_CAND = [(1024,1), (512,2), (256,4), (128,8), (512,1), (256,2), (128,4)]
RETRIEVAL_CAND = [(1024,1)]

class Data:
    @staticmethod
    def _process_split_into_chunks(data, indices, tokenizer, min_length, max_length, window_size, enable_reconstruct):
        outputs = {'input_ids': [], 'attention_mask': [], "labels": [], "length": [], "index": []}

        for i, (input_ids, attention_mask, labels) in enumerate(zip(data['input_ids'], data['attention_mask'], data['labels'])):
            input_ids_raw = torch.tensor(input_ids).reshape(1, 1, -1)
            attention_mask_raw = torch.tensor(attention_mask).reshape(1, 1, -1)
            labels_raw = torch.tensor(labels).reshape(1, 1, -1)
            length = input_ids_raw.shape[-1]

            input_ids, attention_mask, labels = split_into_chunks(
                input_ids=input_ids_raw,
                attention_mask=attention_mask_raw,
                labels=labels_raw,
                window_size=window_size,
                pad_token_id=tokenizer.pad_token_id,
            )

            assert attention_mask is not None, f"attention_mask is None: {attention_mask}"
            assert labels is not None, f"labels is None: {labels}"

            if enable_reconstruct:
                num_chunk = input_ids_raw.shape[-1] // window_size
                input_ids_rec_list = []
                attention_mask_rec_list = []
                labels_rec_list = []
                # generate num_chunk random numbers from 0 to 2 ** 30 - 1
                # random_numbers = torch.randint(0, 2 ** 30, (num_chunk,))
                for i in range(num_chunk):
                    st = random.randint(0, (i + 1) * window_size - 1)
                    ed = min(st + window_size, (i + 1) * window_size)
                    input_ids_rec, attention_mask_rec, labels_rec = input_ids_raw[:, :, st:ed], attention_mask_raw[:, :, st:ed], labels_raw[:, :, st:ed]
                    input_ids_rec = F.pad(input_ids_rec, (0, window_size - input_ids_rec.shape[-1]), value=tokenizer.pad_token_id)
                    attention_mask_rec = F.pad(attention_mask_rec, (0, window_size - attention_mask_rec.shape[-1]), value=0)
                    labels_rec = F.pad(labels_rec, (0, window_size - labels_rec.shape[-1]), value=-100)
                    input_ids_rec_list.append(input_ids_rec)
                    attention_mask_rec_list.append(attention_mask_rec)
                    labels_rec_list.append(labels_rec)
                
                input_ids = torch.cat([input_ids] + input_ids_rec_list, dim=1)
                attention_mask = torch.cat([attention_mask] + attention_mask_rec_list, dim=1)
                labels = torch.cat([labels] + labels_rec_list, dim=1)

            outputs['input_ids'].append(input_ids)
            outputs['attention_mask'].append(attention_mask)
            outputs['labels'].append(labels)
            outputs['length'].append(length)
            outputs['index'].append(indices[i])

        return outputs

    @staticmethod
    def _process_language_modeling(data, indices, tokenizer, min_length, max_length, window_size):

        outputs = {'input_ids': [], 'attention_mask': [], "labels": [], "length": [], "index": []}

        for i, text in enumerate(data['text']):
            # truncate text for faster processing
            encoded = tokenizer(text, return_tensors="pt")
            # if len(encoded["input_ids"]) < min_length:
            #     continue
            # elif len(encoded['input_ids']) < max_length:
            #     encoded = add_eos(encoded, tokenizer.eos_token_id)
            # else:
            #     for k, v in encoded.items():
            #         encoded[k] = v[:max_length]
            if encoded["input_ids"].shape[-1] < min_length:
                continue
            for k, v in encoded.items():
                encoded[k] = v[:, :max_length]

            # encoded["labels"] = encoded["input_ids"].copy()

            # for k, v in encoded.items():
            #     outputs[k].append(v)
            # # length is required for grouping
            # outputs["length"].append(len(encoded['input_ids']))
            # outputs["index"].append(indices[i])
            input_ids = encoded["input_ids"].reshape(1, 1, -1)
            attention_mask = encoded["attention_mask"].reshape(1, 1, -1)
            labels = input_ids
            token_ids, attention_mask, labels = split_into_chunks(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                window_size=window_size,
                pad_token_id=tokenizer.pad_token_id,
            )
            outputs['input_ids'].append(token_ids)
            outputs['attention_mask'].append(attention_mask)
            outputs['labels'].append(labels)
            outputs['length'].append(token_ids.shape[-1])
            outputs['index'].append(indices[i])


        return outputs

    @staticmethod
    def _process_instruction_tuning(data, indices, tokenizer, chat_template, min_length, max_length, window_size, eval_mode=False):
        import torch
        import torch.nn.functional as F

        outputs = {'input_ids': [], 'attention_mask': [], "labels": [], "length": [], "index": []}

        for i, source in enumerate(data['conversations']):
            if len(source) == 0:
                logger.warning(f"Empty conversation found in {data['id'][i]}")
                continue

            if source[0]["role"] != 'user':
                # Skip the first one if it is not from user
                source = source[1:]

            # NOTE: in evaluation, we only use the first turn in the conversation
            if eval_mode:
                # a string (the expected output from the assistant)
                if len(source) > 1:
                    labels = source[1]['content']
                else:
                    labels = None
                source = source[:1]

            encoded = apply_chat_template(
                chat_template, 
                source, 
                tokenizer=tokenizer, 
                # only return labels in evaluation mode
                return_labels=not eval_mode,
                add_generation_prompt=eval_mode, 
            ).encoded
            token_ids = torch.tensor(encoded["input_ids"]).reshape(1, 1, -1)
            attention_mask = torch.tensor(encoded["attention_mask"]).reshape(1, 1, -1)
            labels = torch.tensor(encoded["labels"]).reshape(1, 1, -1)

            if token_ids.numel() > max_length:
                continue
            if token_ids.numel() > window_size:
                token_ids = token_ids[:, :, :window_size]
                attention_mask = attention_mask[:, :, :window_size]
                labels = labels[:, :, :window_size]

            if "context" in data:
                context_text = data["context"][i]
                # truncate text for faster processing
                # context_encoded = tokenizer(context_text, return_tensors="pt")
                context_input_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": context_text}],
                    tokenize=True, return_tensors="pt"
                )
                context_encoded = {
                    "input_ids": context_input_ids,
                    "attention_mask": torch.ones_like(context_input_ids),
                }

                # print(f"context: {context_encoded['input_ids'].shape[-1]}, input: {token_ids.shape[-1]}")

                # if context_encoded["input_ids"].shape[-1] < min_length:
                #     continue
                # for k, v in context_encoded.items():
                #     context_encoded[k] = v[:, :max_length]
                if context_encoded["input_ids"].numel() > max_length:
                    continue

                # # add two eos token before each window
                # if context_encoded["input_ids"][-1] != tokenizer.eos_token_id:
                #     context_encoded["input_ids"].append(tokenizer.eos_token_id)
                #     context_encoded["attention_mask"].append(1)
                # context_encoded["input_ids"].append(tokenizer.eos_token_id)
                # context_encoded["attention_mask"].append(1)
                # context_encoded = add_eos(context_encoded, tokenizer.eos_token_id)
                # context_encoded["labels"] = context_encoded["input_ids"].copy()
                token_ids_context, attention_mask_context, labels_context = split_into_chunks(
                    input_ids=context_encoded["input_ids"].reshape(1, 1, -1),
                    attention_mask=context_encoded["attention_mask"].reshape(1, 1, -1),
                    labels=torch.full_like(context_encoded["input_ids"], -100).reshape(1, 1, -1),
                    window_size=window_size,
                    pad_token_id=tokenizer.pad_token_id,
                )

                # # FIXME: hard code the max length of context
                # if len(context_encoded["input_ids"]) > 1024:
                #     continue

                # assert set(encoded.keys()) == set(context_encoded.keys()), f"Keys of encoded and context_encoded are different: {encoded.keys()} vs {context_encoded.keys()}"
                # for k, v in encoded.items():
                #     encoded[k] =  context_encoded[k] + v

                token_ids, attention_mask, labels = merge_chunks(
                    token_ids_context, attention_mask_context, labels_context,
                    token_ids, attention_mask, labels,
                    window_size=window_size,
                    pad_token_id=tokenizer.pad_token_id,
                )

                # print(token_ids.shape)
            
            # if token_ids shape is larger than max_length, remove
            # if token_ids.numel() > max_length:
            #     continue
            if token_ids.numel() < min_length:
                continue

            outputs['input_ids'].append(token_ids)
            outputs['attention_mask'].append(attention_mask)
            outputs['labels'].append(labels)
            outputs['length'].append(token_ids.shape[-1])
            outputs['index'].append(indices[i])

            # # skip data that not fall in between min_length and max_length
            # if min_length is not None and len(encoded["input_ids"]) < min_length:
            #     continue
            # if max_length is not None and len(encoded["input_ids"]) > max_length:
            #     for k, v in encoded.items():
            #         encoded[k] = v[:max_length]
            
            # if eval_mode:
            #     encoded["labels"] = labels

            # for k, v in encoded.items():
            #     outputs[k].append(v)
            # outputs['length'].append(len(encoded['input_ids']))
            # outputs['index'].append(
            # [i])

        return outputs

    @staticmethod
    def prepare_train_data(data_files=None, tokenizer=None, max_length=4096, min_length=512, window_size=1024, chat_template="vicuna", enable_reconstruct=False, seed=42, cache_dir=None, load_from_cache_file=None):
        if data_files is None:
            return None

        if isinstance(data_files, list):
            logger.info(f"Loading training data from {data_files}...")
        elif isinstance(data_files, str):
            logger.info(f"Loading training data from {data_files}...")
            data_files = [data_files]
        else:
            raise ValueError(f"Invalid training data {data_files}!")

        data_2_num_sample = {}
        for data_file in data_files:
            match = re.search("\[(\d*)\]", data_file)
            if match:
                max_sample_num = int(match.group(1))
                data_file = re.sub("\[(\d*)\]", "", data_file)
            else:
                max_sample_num = None
            data_2_num_sample[data_file] = max_sample_num   
        
        random.seed(seed)
        torch.manual_seed(seed)
        
        train_datasets = []
        # for data_file, max_sample_num in data_2_num_sample.items():
        for data_file in data_files:
            max_sample_num = data_2_num_sample[data_file]

            if os.path.isdir(data_file) and os.path.exists(os.path.join(data_file, "dataset_info.json")):
                # the dataset may be save_to_disk in advance
                dataset = datasets.load_from_disk(data_file)
            
                process_fn = partial(
                    Data._process_split_into_chunks, 
                    tokenizer=tokenizer, 
                    min_length=min_length, 
                    max_length=max_length,
                    window_size=window_size,
                    enable_reconstruct=enable_reconstruct,
                )
                dataset = dataset.map(process_fn, batched=True, num_proc=16, remove_columns=dataset.column_names, batch_size=1024, with_indices=True, load_from_cache_file=load_from_cache_file)

            else:
                # the dataset is a json file
                dataset = datasets.load_dataset('json', data_files=data_file, split='train', cache_dir=cache_dir)

                column_names = dataset.column_names
                if "text" in column_names:
                    process_fn = partial(
                        Data._process_language_modeling, 
                        tokenizer=tokenizer, 
                        min_length=min_length, 
                        max_length=max_length,
                        window_size=window_size,
                    )
                elif "conversations" in column_names:
                    process_fn = partial(
                        Data._process_instruction_tuning, 
                        tokenizer=tokenizer, 
                        chat_template=chat_template, 
                        min_length=min_length, 
                        max_length=max_length,
                        window_size=window_size,
                    )
                else:
                    raise ValueError(f"Found neither 'text' nor 'conversations' in the training data!")

                dataset = dataset.map(process_fn, batched=True, num_proc=16, remove_columns=dataset.column_names, batch_size=64, with_indices=True, load_from_cache_file=load_from_cache_file)

                print(f"dataset: {data_file}, max_length: {max_length}, num samples: {len(dataset)}")

            if max_sample_num is not None and len(dataset) > max_sample_num:
                dataset = dataset.train_test_split(max_sample_num, seed=seed)["test"]

            # index column is useless in training
            if "index" in dataset.column_names:
                dataset = dataset.remove_columns(["index"])

            train_datasets.append(dataset)

        dataset = datasets.concatenate_datasets(train_datasets)

        return dataset

    @staticmethod
    def prepare_eval_data(data_files=None, tokenizer=None, max_length=4096, min_length=512, window_size=1024, chat_template="vicuna", max_eval_num=None, cache_dir=None, seed=42, load_from_cache_file=None):
        if data_files is None:
            return None

        random.seed(seed)
        torch.random.manual_seed(seed)

        if max_eval_num is not None:
            dataset = datasets.load_dataset('json', data_files=data_files, split=f'train[:{max_eval_num}]', cache_dir=cache_dir)
        else:
            dataset = datasets.load_dataset('json', data_files=data_files, split='train', cache_dir=cache_dir)

        column_names = dataset.column_names
        if "text" in column_names:
            process_fn = partial(
                Data._process_language_modeling, 
                tokenizer=tokenizer, 
                min_length=min_length, 
                max_length=max_length,
                window_size=window_size,
            )
        elif "conversations" in column_names:
            process_fn = partial(
                Data._process_instruction_tuning, 
                tokenizer=tokenizer, 
                chat_template=chat_template, 
                min_length=min_length, 
                max_length=max_length,
                eval_mode=True,
                window_size=window_size,
            )
        else:
            raise ValueError(f"Found neither 'text' nor 'conversations' in the training data!")

        dataset = dataset.map(process_fn, batched=True, num_proc=16, remove_columns=dataset.column_names, with_indices=True, load_from_cache_file=load_from_cache_file)
        return dataset