import os
import sys
import pytz
import json
import torch
import shutil
import pathlib
import time
import pickle
import logging
import string
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass
from transformers.tokenization_utils import PreTrainedTokenizer
from datetime import datetime
from collections import OrderedDict
from typing import Optional, List, Dict, Any, Mapping, Iterable, Union
import os
import json
import inspect
import numpy as np
from functools import partial
from rouge import Rouge
from tqdm import tqdm
from transformers.utils import logging

logger = logging.get_logger(__name__)
torch.set_num_threads(1)


@contextmanager
def do_nothing():
    yield

def optional_grad_ctx(with_grad=False):
    if with_grad:
        return do_nothing()
    else:
        return torch.no_grad()

def makedirs(path):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return path

def clear_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def split_file_dir_name_ext(path):
    """Return the directory, name, and extension of a given file."""
    p = pathlib.Path(path)
    assert p.is_file(), f"{path} is not a valid file!"
    return p.parent, p.stem, p.suffix

def save_pickle(obj, path:str):
    """
    Save pickle file.
    """
    if not os.path.exists(path):
        makedirs(path)
    with open(path, "wb") as f:
        return pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def save_json(obj, path:str):
    if not os.path.exists(path):
        makedirs(path)
    with open(path, "w") as f:
        return json.dump(obj, f)

def load_json(path, lines=False):
    if lines:
        output = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                output.append(json.loads(line))
        return output
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def format_numel_str(numel: int) -> str:
    T = 1e12
    B = 1e9
    M = 1e6
    K = 1e3
    if numel >= T:
        return f"{numel / T:.2f} T"
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"

def batched_iter(iterable: Iterable, max_batch_size: int):
    """ Batches an iterable into lists of given maximum size, yielding them one by one. """
    batch = []
    for element in iterable:
        batch.append(element)
        if len(batch) >= max_batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch

def show_time(times):
    times = np.array(times)
    times = np.diff(times, axis=-1)
    print(times)
    return times

@contextmanager
def filelock(path, process_index=0):
    while os.path.exists(path):
        if i == 0 and process_index == 0:
            logger.info("found lock, waiting for other programs...")
        time.sleep(3)
        i = 1
    if process_index == 0:
        save_json("this is a lock", path)
    yield
    if process_index == 0:
        os.remove(path)

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

def wrap_text(s):
    """Capitalize and add punctuation if there isn't."""
    s = s.strip()
    if not s[0].isupper():
        s = s[0].capitalize() + s[1:]
    if s[-1] not in string.punctuation:
        s += "."
    return s

def min_max_normalize(array):
    return (array - array.min(-1)[:,None])/(array.max(-1) - array.min(-1))[:, None]

def softmax(x:np.ndarray, axis=-1):
    if isinstance(x, list):
        x = np.array(x)
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def get_max_length_in_nested_lists(lst):
    if len(lst) and isinstance(lst[0], list):
        lengths = []
        for elem in lst:
            length = get_max_length_in_nested_lists(elem)
            lengths.append(length)
        max_length = max(lengths)
        return max_length
    else:
        return len(lst)

def pad_nested_lists(lst, max_length, padding_value, padding_side="right"):
    if isinstance(lst, list) and len(lst) and isinstance(lst[0], list):
        masks = []
        for i, elem in enumerate(lst):
            lst[i], mask = pad_nested_lists(elem, max_length, padding_value, padding_side)
            masks.append(mask)
        return lst, masks
    elif isinstance(lst, list):
        if padding_side == "right":
            mask = [1] * len(lst) + [0] * (max_length - len(lst))
            lst = lst + [padding_value for _ in range(max_length - len(lst))]
            return lst, mask
        else:
            mask = [0] * (max_length - len(lst)) + [1] * len(lst)
            lst = [padding_value for _ in range(max_length - len(lst))] + lst
            return lst, mask
    else:
        raise NotImplementedError(f"Unrecognized type {lst}")

def mask_nested_lists(lst, mask_target, mask_value=0):
    if isinstance(lst[0], list):
        for i, elem in enumerate(lst):
            lst[i] = mask_nested_lists(elem, mask_target, mask_value)
        return lst
    else:
        return [x if x != mask_target else mask_value for x in lst]

def are_elements_of_same_length(lst: List):
    if not isinstance(lst[0], list):
        return False

    length = len(lst[0])
    return all(len(x) == length if isinstance(x, list) else False for x in lst)

def add_eos(inputs: Mapping, eos_token_id: int):
    """Add eos for BatchEncoding object."""
    assert isinstance(inputs["input_ids"], list), f"Make sure the return_tensors are set to list!"
    if inputs["input_ids"][-1] != eos_token_id:
        for k, v in inputs.items():
            if k in ["input_ids", "labels"]:
                v = v + [eos_token_id]
            elif k == "attention_mask":
                v = v + [1]
            elif k == "position_ids":
                v = v + [v[-1] + 1]
            elif k == "token_type_ids":
                v = v + v[-1:]
            else:
                raise NotImplementedError(f"Inputs key {k} not implemented!")
            inputs[k] = v
    return inputs

def remove_eos(inputs: Mapping, eos_token_ids: Union[List,int]):
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    input_ids = inputs["input_ids"]
    eos_idx = [i for i, x in enumerate(input_ids) if x in eos_token_ids][0]
    for k, v in inputs.items():
        inputs[k].pop(eos_idx)
    return inputs

def mix_parameters(models: List[torch.nn.Module], weights: Optional[List[float]]=None):
    """Mix parameters of different models according to given weights.
    
    Returns:
        the model with mixed parameters.
    """
    new_state_dict = OrderedDict()
    if weights is None:
        weights = [1 / len(models) for _ in range(len(models))]
    else:
        assert len(weights) == len(models), f"Make sure the size of mix weights equals to the number of models!"

    for name_param_pairs in zip(*[model.state_dict().items() for model in models]):
        names = [name_param_pair[0] for name_param_pair in name_param_pairs]
        params = [name_param_pair[1] for name_param_pair in name_param_pairs]

        assert all(name == names[0] for name in names), f"Found incompatible key in {names}!"
        name = names[0]
        mixed_param = None

        # there may be non-float parameters stored, which should not be mixed
        if params[0].dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            assert all((param == params[0]).all() for param in params), f"Found incompatible value in non-float tensor {params}!"
            new_state_dict[name] = params[0]
            continue

        for weight, param in zip(weights, params):
            if mixed_param is None:
                mixed_param = weight * param
            else:
                mixed_param += weight * param
            new_state_dict[name] = mixed_param
            
    model = models[0]
    info = model.load_state_dict(new_state_dict)
    print(info)
    return model


class FileLogger:
    def __init__(self, log_file) -> None:
        self.log_file = log_file
    
    def log(self, metrics, **kwargs):
        with open(self.log_file, "a+") as f:
            # get current time
            tz = pytz.timezone('Asia/Shanghai')
            time = f"{'Time': <10}: {json.dumps(datetime.now(tz).strftime('%Y-%m-%d, %H:%M:%S'), ensure_ascii=False)}\n"
            print(time)
            command = f"{'Command': <10}: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}\n"
            print(command)
            metrics = f"{'Metrics': <10}: {json.dumps(metrics, ensure_ascii=False)}\n"
            msg = time + command

            for key, value in kwargs.items():
                x = f"{key: <10}: {json.dumps(value, ensure_ascii=False)}\n"
                print(x)
                msg += x
            msg += metrics
            print(metrics)
            f.write(str(msg) + "\n")

def batch_chunks(batch_value, padding_value):
    # batch_value: [(1, N1, L1), ...]
    import torch.nn.functional as F
    seq_len = max([x.shape[-1] for x in batch_value])
    # pad to the same length
    batch_value = [F.pad(x, (0, seq_len - x.shape[-1]), value=padding_value) for x in batch_value]
    # concatenate on the first dimension
    return torch.cat(batch_value, dim=0)

def split_into_chunks(input_ids=None, attention_mask=None, labels=None, window_size=1024, pad_token_id=0):
    import torch.nn.functional as F
    number_windows = (input_ids.shape[-1] + window_size - 1) // window_size
    seq_len = (input_ids.shape[-1] + number_windows - 1) // number_windows
    input_ids = F.pad(input_ids, (0, number_windows * seq_len - input_ids.shape[-1]), value=pad_token_id).reshape(-1, number_windows, seq_len)
    if attention_mask is not None:
        attention_mask = F.pad(attention_mask, (0, number_windows * seq_len - attention_mask.shape[-1]), value=0).reshape(-1, number_windows, seq_len)
    if labels is not None:
        labels = F.pad(labels, (0, number_windows * seq_len - labels.shape[-1]), value=-100).reshape(-1, number_windows, seq_len)
    return input_ids, attention_mask, labels

def merge_chunks(input_ids_seq_1, attention_mask_seq_1, label_seq_1, input_ids_seq_2, attention_mask_seq_2, label_seq_2, window_size=1024, pad_token_id=0):
    # input_ids_seq_1: B x N1 x L1
    # input_ids_seq_2: B x N2 x L2
    import torch.nn.functional as F
    assert input_ids_seq_1.shape[1] <= window_size, "input_ids_seq_1 should be less than window_size"
    assert input_ids_seq_2.shape[1] <= window_size, "input_ids_seq_2 should be less than window_size"
    seq_len = max(input_ids_seq_1.shape[2], input_ids_seq_2.shape[2])
    # if None, set to the maximum length
    if attention_mask_seq_1 is None:
        attention_mask_seq_1 = torch.ones_like(input_ids_seq_1)
    if attention_mask_seq_2 is None:
        attention_mask_seq_2 = torch.ones_like(input_ids_seq_2)
    if label_seq_1 is None:
        label_seq_1 = torch.full_like(input_ids_seq_1, -100)
    if label_seq_2 is None:
        label_seq_2 = torch.full_like(input_ids_seq_2, -100)
    # padding if necessary
    if input_ids_seq_1.shape[2] < seq_len:
        input_ids_seq_1 = F.pad(input_ids_seq_1, (0, seq_len - input_ids_seq_1.shape[2]), value=pad_token_id)
        attention_mask_seq_1 = F.pad(attention_mask_seq_1, (0, seq_len - attention_mask_seq_1.shape[2]), value=0)
        label_seq_1 = F.pad(label_seq_1, (0, seq_len - label_seq_1.shape[2]), value=-100)
    if input_ids_seq_2.shape[2] < seq_len:
        input_ids_seq_2 = F.pad(input_ids_seq_2, (0, seq_len - input_ids_seq_2.shape[2]), value=pad_token_id)
        attention_mask_seq_2 = F.pad(attention_mask_seq_2, (0, seq_len - attention_mask_seq_2.shape[2]), value=0)
        label_seq_2 = F.pad(label_seq_2, (0, seq_len - label_seq_2.shape[2]), value=-100)
    # merge chunks
    input_ids = torch.cat([input_ids_seq_1, input_ids_seq_2], dim=1)
    attention_mask = torch.cat([attention_mask_seq_1, attention_mask_seq_2], dim=1)
    labels = torch.cat([label_seq_1, label_seq_2], dim=1)
    return input_ids, attention_mask, labels


@dataclass
class DefaultDataCollator:
    """
    Data collator that can:
    1. Dynamically pad all inputs received. The inputs must be dict of lists.
    2. Add position_ids based on attention_mask if required.
    """

    tokenizer: PreTrainedTokenizer
    attention_padding_value: int = 0
    label_padding_value: int = -100

    keys_to_pad = {"input_ids", "attention_mask", "labels"}
    keys_to_tensorize = {"input_ids", "attention_mask", "labels", "position_ids", "token_type_ids", "length", "depth", "index"}

    def __call__(self, batch_elem: List) -> Dict[str, Any]:
        import torch.nn.functional as F

        first_elem = batch_elem[0]
        return_batch = {}

        for key, value in first_elem.items():
            # HACK: any key containing attention_mask must be attention_mask
            # important to assign different pad token for different types of inputs
            if "attention_mask" in key:
                pad_token_id = self.attention_padding_value
            elif "label" in key:
                pad_token_id = self.label_padding_value
            else:
                pad_token_id = self.tokenizer.pad_token_id

            batch_value = [elem[key] for elem in batch_elem]
            if key in self.keys_to_pad:
                batch_value = [torch.tensor(x) for x in batch_value]
                if batch_value[0].dim() == 3:
                    max_length = max([x.shape[-1] for x in batch_value])
                    batch_value = torch.cat(
                        [F.pad(x, (0, max_length - x.shape[-1]), value=pad_token_id) for x in batch_value],
                        dim=0,
                    )
            elif key in self.keys_to_tensorize:
                batch_value = torch.tensor(batch_value)
            else:
                raise ValueError(f"Key {key} not recognized!")

            return_batch[key] = batch_value

            # batch_value = [elem[key] for elem in batch_elem]
            # pad all lists and nested lists
        #     if isinstance(value, list) and key in self.keys_to_tensorize:
        #         max_length = get_max_length_in_nested_lists(batch_value)
        #         batch_value, _ = pad_nested_lists(batch_value, max_length, pad_token_id, self.tokenizer.padding_side)

        #     if key in self.keys_to_tensorize:
        #         return_batch[key] = torch.tensor(batch_value)
        #     else:
        #         # handle strings and None
        #         return_batch[key] = batch_value
        return return_batch



class Metric:
    """Class for computing metrics and some post-processings."""
    @classmethod
    def get_metric_fn(cls, metrics, **kwds):
        assert isinstance(metrics, list) or isinstance(metrics, tuple), "You must pass metric_names in a list or tuple!"
        return_metrics = {}
        # get all methods
        metric_fns = []

        all_metric_names = [x[0] for x in inspect.getmembers(cls, predicate=inspect.isfunction) if not x[0].startswith("get_")]
        for metric_name in metrics:
            if metric_name in all_metric_names:
                metric_fns.append(partial(getattr(cls, metric_name), **kwds))
            else:
                raise NotImplementedError(f"Metric {metric_name} not implemented!")

        def compute_metrics(*args, **kwargs):
            for metric_fn in metric_fns:
                # call corresponding method
                metric = metric_fn(*args, **kwargs)
                # NOTE: some metric_fn are only used for post-processing and saving results, which return None by default
                if metric is not None:
                    return_metrics.update(metric)
            return return_metrics
        return compute_metrics
    
    def get_save_path(eval_data, output_dir=None, field="result", save_name=None):
        """
        if output_dir is None:
            -> {eval_data_dir}/{eval_data_name}.{field}.{save_name}.{eval_data_ext}
        else:
            -> {output_dir}/{eval_data_name}.{field}.{save_name}.{eval_data_ext}
        """
        eval_data_dir, eval_data_name, eval_data_ext = split_file_dir_name_ext(eval_data)
        if output_dir is None:
            output_dir = eval_data_dir
        fields = [eval_data_name, field]
        if save_name is not None:
            fields.append(save_name)
        save_path = os.path.join(output_dir, ".".join(fields) + eval_data_ext)
        makedirs(save_path)
        return save_path

    def save_result(preds, labels, save_path, indices=None, **kwargs):
        if len(preds) != len(labels):
            logger.warning(f"There are {len(preds)} samples in predictions while {len(labels)} samples in labels!")
            labels = labels[:min(len(preds), len(labels))]
            preds = preds[:min(len(preds), len(labels))]
        
        with open(save_path, "w", encoding="utf-8") as f:
            for i, (pred, label) in enumerate(zip(preds, labels)):
                item = {
                    "prediction": pred,
                    "target": label,
                }
                if indices is not None:
                    item["index"] = indices[i]
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def rouge(preds, labels, **kwargs):
        rouge = Rouge()

        if len(preds) != len(labels):
            logger.warning(f"There are {len(preds)} samples in predictions while {len(labels)} samples in labels!")
            labels = labels[:min(len(preds), len(labels))]
            preds = preds[:min(len(preds), len(labels))]

        preds = normalize_text(preds)
        labels = normalize_text(labels)

        # filter empty preditions
        preds = [":)" if len(pred) == 0 else pred for pred in preds]

        score = rouge.get_scores(preds, labels, avg=True)

        metric = {
            "rouge-1": score["rouge-1"]["f"],
            "rouge-2": score["rouge-2"]["f"],
            "rouge-l": score["rouge-2"]["f"],
        }
        return metric
    
    # def acc(eval_data=None, **kwds):
    #     if eval_data is not None:
    #         data_labels = Metric._prepare_label(eval_data)

    #     def compute_metric(indices, preds, labels=None, **kwargs):
    #         if labels is None:
    #             labels = data_labels

    #         if len(preds) != len(labels):
    #             logger.warning(f"There are {len(preds)} queries in predictions while {len(labels)} queries in labels!")

    #         labels = [labels[query_id] for query_id in indices]

    #         preds = normalize_text(preds)
    #         labels = normalize_text(labels)

    #         overlap = 0
    #         for pred, label in zip(preds, labels):
    #             if pred == label:
    #                 overlap += 1

    #         metric = {
    #             "acc": overlap / len(preds),
    #         }
    #         return metric
    #     return compute_metric
