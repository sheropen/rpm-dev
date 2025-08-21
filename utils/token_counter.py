#!/usr/bin/env python3
"""
Script to count tokens in input text using Mistral-7B-Instruct-v0.2 tokenizer
"""

import argparse
import json
import yaml
import os
import sys
from typing import Dict, List, Any
from transformers import AutoTokenizer

def load_tokenizer(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """Load the Mistral tokenizer"""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def count_tokens(text: str, tokenizer) -> Dict[str, int]:
    """Count tokens in text and return detailed statistics"""
    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=True)
    
    # Get token IDs and decoded tokens for analysis
    token_ids = tokens
    decoded_tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
    
    # Count special tokens
    special_tokens = sum(1 for token_id in token_ids if tokenizer.convert_ids_to_tokens(token_id).startswith('<'))
    
    # Count regular tokens
    regular_tokens = len(token_ids) - special_tokens
    
    return {
        'total_tokens': len(token_ids),
        'regular_tokens': regular_tokens,
        'special_tokens': special_tokens,
        'token_ids': token_ids,
        'decoded_tokens': decoded_tokens
    }

def analyze_text_file(file_path: str, tokenizer) -> Dict[str, Any]:
    """Analyze a text file and return token statistics"""
    print(f"Analyzing file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    stats = count_tokens(content, tokenizer)
    
    # Add file info
    stats['file_path'] = file_path
    stats['file_size_bytes'] = len(content.encode('utf-8'))
    stats['file_size_chars'] = len(content)
    stats['file_size_lines'] = len(content.split('\n'))
    
    return stats

def analyze_html_file(file_path: str, tokenizer) -> Dict[str, Any]:
    """Analyze an HTML file and return token statistics"""
    print(f"Analyzing HTML file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Basic HTML cleaning (remove extra whitespace)
    import re
    cleaned_content = re.sub(r'\s+', ' ', content).strip()
    
    stats = count_tokens(cleaned_content, tokenizer)
    
    # Add file info
    stats['file_path'] = file_path
    stats['file_size_bytes'] = len(content.encode('utf-8'))
    stats['file_size_chars'] = len(content)
    stats['file_size_chars_cleaned'] = len(cleaned_content)
    stats['file_size_lines'] = len(content.split('\n'))
    
    return stats

def analyze_prompt_template(prompts_file: str, tokenizer) -> Dict[str, Any]:
    """Analyze a prompt template from YAML file"""
    print(f"Analyzing prompt template: {prompts_file}")
    
    with open(prompts_file, 'r') as f:
        prompts = yaml.safe_load(f)
    
    prompt_template = prompts['USER_PROMPT']
    stats = count_tokens(prompt_template, tokenizer)
    
    # Add file info
    stats['file_path'] = prompts_file
    stats['file_size_chars'] = len(prompt_template)
    stats['file_size_lines'] = len(prompt_template.split('\n'))
    
    return stats

def analyze_formatted_prompt(prompt_template: str, sample: Dict[str, Any], content: str, task_type: str, tokenizer) -> Dict[str, Any]:
    """Analyze a formatted prompt with sample data"""
    print(f"Analyzing formatted prompt for task: {task_type}")
    
    if task_type == "html_to_tsv":
        # Extract fields from sample for html_to_tsv task
        task_topic = sample.get('task_topic', '')
        task_description = sample.get('task_description', '')
        tsv_header = sample.get('tsv_header', '')
        filtering_instruction = sample.get('filtering_instruction', '')
        
        # Format the prompt
        formatted_prompt = prompt_template.format(
            html_str=content,
            task_topic=task_topic,
            task_description=task_description,
            tsv_header=tsv_header,
            filtering_instruction=filtering_instruction
        )
    elif task_type == "pseudo_to_code":
        # For pseudo_to_code task, just append the pseudocode content
        formatted_prompt = prompt_template + "\n\n" + content
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    stats = count_tokens(formatted_prompt, tokenizer)
    
    # Add metadata
    stats['task_type'] = task_type
    stats['task_id'] = sample.get('task_id', 'unknown')
    stats['content_length'] = len(content)
    stats['formatted_prompt_length'] = len(formatted_prompt)
    
    return stats

def print_token_analysis(stats: Dict[str, Any], show_tokens: bool = False):
    """Print token analysis results"""
    print("\n" + "="*60)
    print("📊 TOKEN ANALYSIS RESULTS")
    print("="*60)
    
    # File information
    if 'file_path' in stats:
        print(f"📁 File: {stats['file_path']}")
        print(f"📏 File size: {stats.get('file_size_bytes', 0):,} bytes")
        print(f"📝 Characters: {stats.get('file_size_chars', 0):,}")
        print(f"📄 Lines: {stats.get('file_size_lines', 0):,}")
    
    # Token information
    print(f"\n🔢 Token Counts:")
    print(f"   Total tokens: {stats['total_tokens']:,}")
    print(f"   Regular tokens: {stats['regular_tokens']:,}")
    print(f"   Special tokens: {stats['special_tokens']:,}")
    
    # Task information
    if 'task_type' in stats:
        print(f"\n🎯 Task Information:")
        print(f"   Task type: {stats['task_type']}")
        print(f"   Task ID: {stats['task_id']}")
        print(f"   Content length: {stats['content_length']:,} chars")
        print(f"   Formatted prompt length: {stats['formatted_prompt_length']:,} chars")
    
    # Efficiency metrics
    if 'file_size_chars' in stats:
        chars_per_token = stats['file_size_chars'] / stats['total_tokens']
        print(f"\n📈 Efficiency Metrics:")
        print(f"   Characters per token: {chars_per_token:.2f}")
        print(f"   Tokens per character: {1/chars_per_token:.2f}")
    
    # Show first few tokens if requested
    if show_tokens and 'decoded_tokens' in stats:
        print(f"\n🔍 First 20 tokens:")
        for i, token in enumerate(stats['decoded_tokens'][:20]):
            print(f"   {i:2d}: '{token}' (ID: {stats['token_ids'][i]})")
        
        if len(stats['decoded_tokens']) > 20:
            print(f"   ... and {len(stats['decoded_tokens']) - 20} more tokens")
    
    print("="*60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Count tokens in text using Mistral-7B-Instruct-v0.2 tokenizer')
    parser.add_argument('--input', '-i', type=str, help='Input text to analyze')
    parser.add_argument('--file', '-f', type=str, help='Input file to analyze')
    parser.add_argument('--html', type=str, help='HTML file to analyze')
    parser.add_argument('--prompts', type=str, help='YAML prompts file to analyze')
    parser.add_argument('--task', type=str, choices=['html_to_tsv', 'pseudo_to_code'], 
                       help='Task type for formatted prompt analysis')
    parser.add_argument('--dataset', type=str, help='Dataset file for sample data')
    parser.add_argument('--sample-index', type=int, default=0, help='Sample index to analyze (default: 0)')
    parser.add_argument('--show-tokens', action='store_true', help='Show individual tokens')
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2',
                       help='Model name for tokenizer')
    
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer = load_tokenizer(args.model)
    
    # Analyze based on input type
    if args.input:
        print(f"Analyzing input text: {args.input[:50]}...")
        stats = count_tokens(args.input, tokenizer)
        print_token_analysis(stats, args.show_tokens)
    
    elif args.file:
        stats = analyze_text_file(args.file, tokenizer)
        print_token_analysis(stats, args.show_tokens)
    
    elif args.html:
        stats = analyze_html_file(args.html, tokenizer)
        print_token_analysis(stats, args.show_tokens)
    
    elif args.prompts:
        stats = analyze_prompt_template(args.prompts, tokenizer)
        print_token_analysis(stats, args.show_tokens)
    
    elif args.task and args.dataset:
        # Load dataset and prompts
        with open(args.dataset, 'r') as f:
            dataset = json.load(f)
        
        with open(args.prompts, 'r') as f:
            prompts = yaml.safe_load(f)
        
        prompt_template = prompts['USER_PROMPT']
        sample = dataset[args.sample_index]
        
        # Get content based on task type
        if args.task == "html_to_tsv":
            html_path = sample['html_path']
            full_path = os.path.join("LongProc/data/html_to_tsv", html_path)
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif args.task == "pseudo_to_code":
            content = sample.get('pseudocode', '')
        else:
            raise ValueError(f"Unsupported task type: {args.task}")
        
        stats = analyze_formatted_prompt(prompt_template, sample, content, args.task, tokenizer)
        print_token_analysis(stats, args.show_tokens)
    
    else:
        print("Please provide input using one of the following options:")
        print("  --input TEXT     : Analyze text directly")
        print("  --file PATH      : Analyze a text file")
        print("  --html PATH      : Analyze an HTML file")
        print("  --prompts PATH   : Analyze a YAML prompts file")
        print("  --task TYPE --dataset PATH : Analyze formatted prompt for a task")
        print("\nExamples:")
        print("  python token_counter.py --input 'Hello world'")
        print("  python token_counter.py --file myfile.txt")
        print("  python token_counter.py --html webpage.html")
        print("  python token_counter.py --task html_to_tsv --dataset LongProc/data/html_to_tsv/html_to_tsv_0.5k.json --prompts LongProc/data/html_to_tsv/prompts.yaml")

if __name__ == "__main__":
    main()

