import sys
sys.path.append('/home/xiangyu/rpm-dev/generative-adapter/src')

# Import functions to load the model, create an adaptor for LoRA weights,
# and perform conditional text generation.

from fastlora.config import FastLoraConfig
from fastlora.model import FastLoraModelForCausalLM, FastLoraModel, get_peft_model_state_dict, set_peft_model_state_dict, load_pretrained_model
import peft.peft_model as peft_model
import peft.mapping as peft_mapping

## monkey patching
peft_model.PEFT_TYPE_TO_MODEL_MAPPING.update({"FASTLORA": FastLoraModel})
peft_mapping.PEFT_TYPE_TO_CONFIG_MAPPING.update({"FASTLORA": FastLoraConfig})
peft_model.get_peft_model_state_dict = get_peft_model_state_dict
peft_model.set_peft_model_state_dict = set_peft_model_state_dict

from fastlora.eval_utils import load_model_and_tokenizer
from fastlora.eval_utils import fastlora_generate_adaptor
from fastlora.eval_utils import fastlora_conditional_generate
from pathlib import Path
import json
import torch
from peft.config import PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the path to the pretrained model checkpoint and the device to run on.
model_name_or_path = "generative-adaptor/Generative-Adapter-Mistral-7B-Instruct-v0.2"
device = 'cuda'
torch_dtype = torch.bfloat16
attn_implementation = 'sdpa'

print("Loading peft config...")

# Load the PEFT configuration from the given pretrained model directory.
peft_config = PeftConfig.from_pretrained(model_name_or_path)

# Get the base model path from the configuration.
base_model_path = peft_config.base_model_name_or_path

# Ensure that the base model path is available.
assert base_model_path is not None, "base_model_name_or_path should not be None"

print("Loading base model...")

# Load the base causal language model from the retrieved base model path.
# The model is loaded with a specific data type and attention implementation.
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch_dtype,
    attn_implementation=attn_implementation,
)


print("Loading FastLora model...")

# Load the FastLora model for causal language modeling.
# This model is built on top of the base model, uses the adapter settings from the PEFT config,
# is not set for training, and is moved to the GPU.
model = FastLoraModelForCausalLM.from_pretrained(
    base_model,
    model_name_or_path,
    adapter_name='default',
    is_trainable=False,
    config=peft_config,
).cuda()


print("Loading tokenizer...")

# Load the tokenizer from the pretrained model directory.
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
print("Tokenizer loaded.")

# Define a prompt prefix that supplies context about a speaker's background.
# This information may influence how the model generates the response.
prompt_prefix = (
    "I volunteer in my spare time. I have been volunteering for 7 years. I volunteer in a homeless shelter in my town. "
    "I'm not into cars. I wrestle for my day job. I like wrestling. I am not super into wrestling. I like crowds and meeting people. "
    "I work out a few times each week when I need to be alone. I like country music a little bit. I like Taylor Swift. "
    "I've lost fights recently. I work out a few times a week. I do not like working on cars. I am not patient. "
    "I have two dogs: Baron Zemo and Spike. I have two older mustangs. I like vintage cars. I'm working on two Mustangs: a 68 and a 66 Hertz clone. "
    "I've been working on cars since 1989. I have a Mustang convertible. I work on my car after work. I get frustrated working on my car sometimes. "
    "I don't like crowds. I like working out. I like classic country. I am a dog trainer. My work keeps me busy."
)

# Define the input prompt that asks a question related to music.
prompt_input = "Hey, remember that time we talked about music? What was the artist you mentioned you could get into?"

# Set parameters for generating the LoRA weights and the text output.
merge_strategy = 'concat'
window_size = 1024
max_new_tokens = 100
stop = ["\n"]

print("Generating LoRA weights...")

# Generate LoRA weights using the model, tokenizer, and prompt prefix.
# This function adapts the model weights based on the given prompt context.
lora_weights = fastlora_generate_adaptor(
    model, tokenizer, 
    prompt_prefix, 
    merge_strategy=merge_strategy, 
    max_window_size=window_size,
)

# Output the number of LoRA weights generated.
print("Number of LoRA weights generated:", len(lora_weights))

print("Generating text...")

# Generate text using the model with the generated LoRA weights.
# The function takes the input prompt and uses chat-style generation.
output_text = fastlora_conditional_generate(
    model, tokenizer, 
    input_text=prompt_input, 
    use_chat=True,
    mode="weights", 
    lora_weights=lora_weights, 
    max_new_tokens=max_new_tokens,
    stop=stop,
)

# Print the generated text.
print(output_text)