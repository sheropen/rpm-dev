# input: a file of test cases
# model: a huggingface model
# process each test case by taking prompt as input and save the output
# output: a file of outputs

## vLLM example BEGIN
# from vllm import LLM, SamplingParams

# # Sample prompts.
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
# # Create a sampling params object.
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# # Create an LLM.
# llm = LLM(model="facebook/opt-125m")
# # Generate texts from the prompts. The output is a list of RequestOutput objects
# # that contain the prompt, generated text, and other information.
# outputs = llm.generate(prompts, sampling_params)
# # Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
## vLLM example END

from vllm import LLM, SamplingParams
import json
from pathlib import Path

def main(args):
    with open(args.input, 'r') as f:
        assert args.input.endswith('.json')
        data_list = json.load(f)
    if args.first_n is not None:
        data_list = data_list[:args.first_n]
    llm = LLM(
        model=args.model,
        dtype="auto",
        max_model_len=32768,
    )
    tokenizer = llm.get_tokenizer()
    if args.format == 'default':
        prompts = [data['prompt'] for data in data_list]
        prompt_token_ids = tokenizer(prompts)["input_ids"]
    elif args.format == 'chat':
        prompt_token_ids = [
            tokenizer.apply_chat_template([
                {"role": "user", "content": data['prompt']}
            ], tokenize=True, add_generation_prompt=True)
            for data in data_list
        ]
    else:
        raise ValueError(f"Unknown format: {args.format}")
    # greedy decoding
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100, stop=["\n"])
    outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
    for data, output in zip(data_list, outputs):
        data['output'] = output.outputs[0].text
        data['prompt'] = tokenizer.decode(output.prompt_token_ids)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(data_list, f, indent=2)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--first_n', type=int, default=None)
    parser.add_argument('--format', type=str, default='default')
    args = parser.parse_args()
    main(args)