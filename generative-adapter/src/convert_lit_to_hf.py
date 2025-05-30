# input: checkpoint_dir

import logging

import torch
from pathlib import Path

from functools import partial
from pathlib import Path
from pprint import pprint

import torch

from litgpt import Config
from litgpt.scripts.convert_hf_checkpoint import layer_template, load_param
from litgpt.utils import (
    extend_checkpoint_dir,
    lazy_load
)
from litgpt.scripts.convert_lit_checkpoint import (
    check_conversion_supported,
    copy_weights_falcon,
    copy_weights_gpt_neox,
    copy_weights_llama,
    copy_weights_phi,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def convert_lit_checkpoint(checkpoint_dir: Path) -> None:
    """Convert a LitGPT trained checkpoint into a Hugging Face Transformers checkpoint."""
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    pprint(locals())

    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    # output_dir.mkdir(parents=True, exist_ok=True)
    # output_path = output_dir / "model.pth"

    if "falcon" in config.name:
        copy_fn = partial(copy_weights_falcon, config.name)
    elif config.mlp_class_name in ("LLaMAMLP", "GemmaMLP", "LLaMAMoE"):
        untie_weights = "Gemma" in config.name
        copy_fn = partial(copy_weights_llama, config, untie_weights=untie_weights)
    elif "phi" in config.name:
        copy_fn = partial(copy_weights_phi, config)
    else:
        copy_fn = copy_weights_gpt_neox

    # initialize a new empty state dict to hold our new weights
    state_dict = {}
    # with incremental_save(output_path) as saver:
    lit_weights = lazy_load(checkpoint_dir / "lit_model.pth")
    lit_weights = lit_weights.get("model", lit_weights)
    check_conversion_supported(lit_weights)
    # copy_fn(sd, lit_weights, saver=saver)
    copy_fn(state_dict, lit_weights)
    # gc.collect()
    # saver.save(sd)
    return state_dict

def main(args):
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = convert_lit_checkpoint(checkpoint_dir)
    config = AutoConfig.from_pretrained(checkpoint_dir / "config.json")
    model = AutoModelForCausalLM.from_config(config, torch_dtype=config.torch_dtype)
    # def get_model(state_dict):
    #     return {k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")}
    # model.load_state_dict(get_model(state_dict))
    model.load_state_dict(state_dict)
    model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    main(args)
