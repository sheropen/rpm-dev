#!/usr/bin/env python3
"""
MetaICL Evaluation Script for FastLoRA

This script evaluates FastLoRA models on MetaICL datasets using the following approach:
1. Use demonstration examples to generate LoRA weights (not in input prompt)
2. Use the adapted model to predict answers for test examples
3. Compare with standard ICL approach
4. Evaluate on 1, 2, 4, 8, 16 demonstrations with 5 random seeds each
"""

import sys
import traceback

# append metaicl to path
sys.path.append('/home/xiangyu/rpm-dev/MetaICL')

import os
import json
import torch
import numpy as np
import random
import logging
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Tuple, Any
import argparse

# Add MetaICL to path
sys.path.append('/home/xiangyu/rpm-dev/MetaICL')

# FastLoRA imports
sys.path.append('/home/xiangyu/rpm-dev/generative-adapter/src')
from fastlora.config import FastLoraConfig
from fastlora.model import FastLoraModelForCausalLM, FastLoraModel, get_peft_model_state_dict, set_peft_model_state_dict
import peft.peft_model as peft_model
import peft.mapping as peft_mapping

# Monkey patching for FastLoRA
peft_model.PEFT_TYPE_TO_MODEL_MAPPING.update({"FASTLORA": FastLoraModel})
peft_mapping.PEFT_TYPE_TO_CONFIG_MAPPING.update({"FASTLORA": FastLoraConfig})
peft_model.get_peft_model_state_dict = get_peft_model_state_dict
peft_model.set_peft_model_state_dict = set_peft_model_state_dict

from fastlora.eval_utils import fastlora_generate_adaptor, fastlora_conditional_generate
from peft.config import PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# MetaICL imports
from utils.data import load_data

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastLoRAMetaICLEvaluator:
    """Evaluator for FastLoRA models on MetaICL tasks"""
    
    def __init__(self, model_path: str, device: str = 'cuda', torch_dtype=torch.bfloat16):
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = None
        self.tokenizer = None
        self.base_model = None
        
        # FastLoRA parameters
        self.merge_strategy = 'concat'
        self.window_size = 1024
        self.max_new_tokens = 50
        self.stop = ["\n", ".", "!", "?"]
        
        self._load_model()
    
    def _load_model(self):
        """Load FastLoRA model and tokenizer"""
        logger.info("Loading FastLoRA model...")
        
        # Load PEFT config
        peft_config = PeftConfig.from_pretrained(self.model_path)
        base_model_path = peft_config.base_model_name_or_path
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=self.torch_dtype,
            attn_implementation='sdpa',
        )
        
        # Load FastLoRA model
        self.model = FastLoraModelForCausalLM.from_pretrained(
            self.base_model,
            self.model_path,
            adapter_name='default',
            is_trainable=False,
            config=peft_config,
        ).to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Model loaded successfully")
    
    def format_demonstrations(self, train_data: List[Dict]) -> str:
        """Format demonstration examples for LoRA weight generation"""
        demonstrations = []
        for example in train_data:
            demo = f"Input: {example['input']}\nOutput: {example['output']}"
            demonstrations.append(demo)
        return "\n\n".join(demonstrations)
    

    
    def compute_option_probability(self, input_text: str, option: str, lora_weights=None) -> float:
        """Compute probability of a specific option given input text"""
        
        # Format the complete sequence
        if input_text.strip().endswith("?"):
            prompt = f"{input_text.strip()} {option.strip()}"
        else:
            prompt = f"{input_text.strip()}\nAnswer: {option.strip()}"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get option tokens (tokens corresponding to the answer)
        input_only = self.tokenizer(input_text.strip(), return_tensors="pt", truncation=True, max_length=1024)
        input_length = input_only['input_ids'].shape[1]
        
        # Apply LoRA weights if provided
        if lora_weights is not None:
            # Store original weights
            original_state = self.model.get_adapter_state_dict('default')
            
            # Apply new LoRA weights
            self.model.load_adapter_state_dict(lora_weights, 'default')
        
        try:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Calculate probability for the option tokens
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Get probabilities for the answer part only
                answer_log_probs = []
                for i in range(input_length, input_ids.shape[1]):
                    if i < input_ids.shape[1] - 1:  # Don't include the last token
                        token_id = input_ids[0, i + 1]  # Next token
                        token_log_prob = log_probs[0, i, token_id]
                        answer_log_probs.append(token_log_prob.item())
                
                # Average log probability
                if len(answer_log_probs) > 0:
                    avg_log_prob = sum(answer_log_probs) / len(answer_log_probs)
                    probability = np.exp(avg_log_prob)
                else:
                    probability = 0.0
                    
        finally:
            # Restore original weights if LoRA weights were applied
            if lora_weights is not None:
                self.model.load_adapter_state_dict(original_state, 'default')
        
        return probability
    
    def predict_with_lora(self, test_data: List[Dict], train_data: List[Dict]) -> List[str]:
        """Make predictions using LoRA-adapted model with generate_answer_fastlora approach"""
        
        # Generate LoRA weights from demonstrations
        if len(train_data) > 0:
            demonstrations = self.format_demonstrations(train_data)
            logger.debug("Generating LoRA weights from demonstrations...")
            lora_weights = fastlora_generate_adaptor(
                model=self.model,
                tokenizer=self.tokenizer,
                context_text=demonstrations,
                merge_strategy=self.merge_strategy,
                max_window_size=self.window_size
            )
            logger.debug(f"Generated {len(lora_weights)} LoRA weight tensors")
        else:
            lora_weights = None
        
        predictions = []
        
        for example in test_data:
            input_text = example['input']
            options = example.get('options', [])
            
            # If it's a classification task with options, use option selection
            if options:
                # For classification tasks, we'll generate text and match to closest option
                try:
                    # Generate answer using FastLoRA
                    generated_answer = fastlora_conditional_generate(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        input_text=input_text,
                        mode="weights",
                        lora_weights=lora_weights,
                        use_chat=True,
                        max_new_tokens=self.max_new_tokens,
                        stop=self.stop
                    )
                    
                    # Match generated answer to closest option
                    best_option = self._match_to_option(generated_answer.strip(), options)
                    predictions.append(best_option)
                    
                    logger.debug(f"Input: {input_text[:100]}...")
                    logger.debug(f"Options: {options}")
                    logger.debug(f"Generated: {generated_answer}")
                    logger.debug(f"Matched to: {best_option}")
                    
                except Exception as e:
                    logger.error(f"Error generating answer with FastLoRA: {e}")
                    logger.error(traceback.format_exc())
                    # Fallback to empty string
                    predictions.append("")
            else:
                # For non-classification tasks, generate text directly
                try:
                    generated_answer = fastlora_conditional_generate(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        input_text=input_text,
                        mode="weights",
                        lora_weights=lora_weights,
                        use_chat=True,
                        max_new_tokens=self.max_new_tokens,
                        stop=self.stop
                    )
                    
                    predictions.append(generated_answer.strip())
                    
                    logger.debug(f"Input: {input_text[:100]}...")
                    logger.debug(f"Generated: {generated_answer}")
                    
                except Exception as e:
                    logger.error(f"Error generating answer with FastLoRA: {e}")
                    logger.error(traceback.format_exc())
                    predictions.append("")
        
        return predictions
    
    def _match_to_option(self, generated_text: str, options: List[str]) -> str:
        """Match generated text to the closest option"""
        generated_lower = generated_text.lower().strip()
        
        # First try exact match
        for option in options:
            if generated_lower == option.lower().strip():
                return option
        
        # Then try substring match
        for option in options:
            if generated_lower in option.lower() or option.lower() in generated_lower:
                return option
        
        # If no match found, return the first option
        logger.debug(f"No match found for '{generated_text}' in options {options}, defaulting to first option")
        return options[0] if options else generated_text
    
    def predict_standard_icl(self, test_data: List[Dict], train_data: List[Dict]) -> List[str]:
        """Make predictions using standard in-context learning (demonstrations in prompt)"""
        predictions = []
        
        for example in test_data:
            input_text = example['input']
            options = example['options']
            
            # Format with demonstrations in prompt
            if len(train_data) > 0:
                demonstrations = self.format_demonstrations(train_data)
                full_prompt = f"{demonstrations}\n\nInput: {input_text}"
            else:
                full_prompt = f"Input: {input_text}"
            
            # Compute probability for each option (without LoRA adaptation)
            option_probs = {}
            for option in options:
                prob = self.compute_option_probability(full_prompt, option, lora_weights=None)
                option_probs[option] = prob
            
            # Select option with highest probability
            best_option = max(option_probs.keys(), key=lambda x: option_probs[x])
            predictions.append(best_option)
        
        return predictions
    
    def evaluate_accuracy(self, predictions: List[str], ground_truths: List[str]) -> float:
        """Calculate accuracy"""
        correct = sum(1 for pred, gt in zip(predictions, ground_truths) if pred.strip() == gt.strip())
        return correct / len(predictions) if len(predictions) > 0 else 0.0
    
    def evaluate_task(self, task_name: str, k_shots: int, seed: int, method: str = 'fastlora') -> Dict:
        """Evaluate on a single task with k demonstrations and specific seed"""
        
        logger.info(f"Evaluating {task_name} with {k_shots} shots, seed {seed}, method {method}")
        
        # Load data directly from files (all files use 16 shots in naming)
        data_dir = '/home/xiangyu/rpm-dev/MetaICL/data'
        task_dir = os.path.join(data_dir, task_name)
        
        # Load train data from 16-shot file
        train_file = os.path.join(task_dir, f"{task_name}_16_{seed}_train.jsonl")
        if not os.path.exists(train_file):
            logger.error(f"Train file not found: {train_file}")
            return None
            
        all_train_data = []
        with open(train_file, 'r') as f:
            for line in f:
                all_train_data.append(json.loads(line))
        
        # Load test data from 16-shot file
        test_file = os.path.join(task_dir, f"{task_name}_16_{seed}_test.jsonl")
        if not os.path.exists(test_file):
            logger.error(f"Test file not found: {test_file}")
            return None
            
        test_data = []
        with open(test_file, 'r') as f:
            for line in f:
                test_data.append(json.loads(line))
        
        logger.info(f"Loaded {len(all_train_data)} training examples and {len(test_data)} test examples")
        
        # Filter to single task
        all_train_data = [dp for dp in all_train_data if dp["task"] == task_name]
        test_data = [dp for dp in test_data if dp["task"] == task_name]
        
        logger.info(f"After filtering by task: {len(all_train_data)} training examples, {len(test_data)} test examples")
        
        # Sample k_shots demonstrations from training data
        if k_shots > len(all_train_data):
            logger.warning(f"Requested {k_shots} shots but only {len(all_train_data)} available. Using all available.")
            train_data = all_train_data
        else:
            # Use the same seed for reproducible sampling
            random.seed(seed)
            train_data = random.sample(all_train_data, k_shots)
            logger.info(f"Sampled {len(train_data)} demonstrations from {len(all_train_data)} available")
        
        # Limit test examples to 16 for faster evaluation
        if len(test_data) > 16:
            random.seed(seed + 1000)  # Different seed for test sampling
            test_data = random.sample(test_data, 16)
            logger.info(f"Sampled {len(test_data)} test examples for evaluation")
        
        if len(test_data) == 0:
            logger.warning(f"No test data found for task {task_name}")
            return None
        
        # Get task configuration
        config_file = f"/home/xiangyu/rpm-dev/MetaICL/config/tasks/{task_name}.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            is_classification = config.get("task_type") == "classification"
            logger.info(f"Task type: {'classification' if is_classification else 'non-classification'}")
        else:
            # Assume classification if config not found
            is_classification = True
            logger.warning(f"Config file not found for {task_name}, assuming classification task")
        
        # Log demonstration and test examples
        logger.info(f"Using {len(train_data)} demonstrations:")
        for i, demo in enumerate(train_data[:3]):  # Log first 3 demonstrations
            logger.info(f"  Demo {i+1}: Input: {demo['input'][:100]}... -> Output: {demo['output']}")
        if len(train_data) > 3:
            logger.info(f"  ... and {len(train_data) - 3} more demonstrations")
        
        # Make predictions
        logger.info(f"Making predictions using method: {method}")
        if method == 'fastlora':
            predictions = self.predict_with_lora(test_data, train_data)
        elif method == 'icl':
            predictions = self.predict_standard_icl(test_data, train_data)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Extract ground truths
        ground_truths = [example['output'] for example in test_data]
        
        # Calculate accuracy
        accuracy = self.evaluate_accuracy(predictions, ground_truths)
        
        # Log some prediction examples
        logger.info(f"Sample predictions:")
        for i in range(min(3, len(predictions))):
            correct = "✓" if predictions[i].strip() == ground_truths[i].strip() else "✗"
            logger.info(f"  {correct} Input: {test_data[i]['input'][:80]}...")
            logger.info(f"    Predicted: {predictions[i]} | Ground Truth: {ground_truths[i]}")
        
        result = {
            'task': task_name,
            'k_shots': k_shots,
            'seed': seed,
            'method': method,
            'accuracy': accuracy,
            'is_classification': is_classification,
            'num_test_examples': len(test_data),
            'num_demonstrations': len(train_data),
            'total_available_train': len(all_train_data)
        }
        
        logger.info(f"Task: {task_name}, K: {k_shots}, Seed: {seed}, Method: {method}, Accuracy: {accuracy:.3f} ({int(accuracy * len(test_data))}/{len(test_data)})")
        
        return result

def main():
    parser = argparse.ArgumentParser(description="Evaluate FastLoRA on MetaICL tasks")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to FastLoRA model")
    parser.add_argument("--tasks", type=str, nargs="+", 
                       default=["financial_phrasebank", "poem_sentiment", "tweet_eval-sentiment"],
                       help="List of tasks to evaluate")
    parser.add_argument("--k_shots", type=int, nargs="+", 
                       default=[1, 2, 4, 8, 16],
                       help="Number of demonstrations to use")
    parser.add_argument("--seeds", type=int, nargs="+",
                       default=[100, 13, 21, 42, 87],
                       help="Random seeds for sampling")
    parser.add_argument("--methods", type=str, nargs="+",
                       default=["fastlora", "icl"],
                       help="Methods to compare (fastlora, icl)")
    parser.add_argument("--output_dir", type=str, default="./results/metaicl_eval",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = FastLoRAMetaICLEvaluator(
        model_path=args.model_path,
        device=args.device
    )
    
    # Run evaluation
    all_results = []
    
    for task in args.tasks:
        for method in args.methods:
            for k in args.k_shots:
                for seed in args.seeds:
                    try:
                        result = evaluator.evaluate_task(task, k, seed, method)

                        if result is not None:
                            all_results.append(result)
                    except Exception as e:
                        logger.error(f"Error evaluating {task} with {k} shots, seed {seed}, method {method}: {e}")
                        logger.error(traceback.format_exc())
                        continue
    
    # Save detailed results
    results_file = Path(args.output_dir) / "detailed_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Compute and save summary statistics
    summary = compute_summary_statistics(all_results)
    summary_file = Path(args.output_dir) / "summary_results.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print_summary(summary)
    
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")

def compute_summary_statistics(results: List[Dict]) -> Dict:
    """Compute summary statistics organized by method, k-shots, individual tasks, and task types."""
    
    summary = {
        'by_method': defaultdict(list),
        'by_k_shots': defaultdict(list),
        'by_task': defaultdict(list),
        'by_task_type': defaultdict(list)
    }
    
    # Group results
    for result in results:
        method = result['method']
        k = result['k_shots']
        task_name = result['task']
        is_classification = result['is_classification']
        task_type = 'classification' if is_classification else 'non_classification'
        accuracy = result['accuracy']
        
        # Group by method
        summary['by_method'][method].append(accuracy)
        
        # Group by method and k
        method_k_key = f"{method}_k{k}"
        summary['by_k_shots'][method_k_key].append(accuracy)
        
        # Group by method, task, and k
        method_task_k_key = f"{method}_{task_name}_k{k}"
        summary['by_task'][method_task_k_key].append(accuracy)
        
        # Group by task type, method, and k
        task_type_method_k_key = f"{task_type}_{method}_k{k}"
        summary['by_task_type'][task_type_method_k_key].append(accuracy)
    
    # Compute statistics for each group
    def compute_stats(acc_list):
        return {
            'mean': np.mean(acc_list),
            'std': np.std(acc_list),
            'count': len(acc_list),
            'min': np.min(acc_list),
            'max': np.max(acc_list)
        }
    
    # Replace lists with summary stats
    for group_key in ['by_method', 'by_k_shots', 'by_task', 'by_task_type']:
        for key, acc_list in summary[group_key].items():
            summary[group_key][key] = compute_stats(acc_list)
    
    return summary

def print_summary(summary: Dict):
    """Print summary statistics in a readable format"""
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    # Find all methods, k values, and tasks
    methods = set()
    k_values = set()
    tasks = set()
    
    for key in summary['by_task'].keys():
        if '_k' in key:
            parts = key.split('_k')
            method_task = parts[0]
            k = int(parts[1])
            
            # Extract method and task from method_task
            method_parts = method_task.split('_')
            method = method_parts[0]
            task = '_'.join(method_parts[1:])
            
            methods.add(method)
            k_values.add(k)
            tasks.add(task)
    
    methods = sorted(methods)
    k_values = sorted(k_values)
    tasks = sorted(tasks)
    
    # Print overall results by method
    print("\nOVERALL RESULTS BY METHOD:")
    print("-" * 40)
    for method in methods:
        if method in summary['by_method']:
            stats = summary['by_method'][method]
            print(f"{method:12}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
    
    # Print results by individual task
    print("\nRESULTS BY TASK:")
    print("-" * 40)
    for task in tasks:
        print(f"\n{task.upper()}:")
        for method in methods:
            # Find all k-shot results for this method-task combination
            task_results = []
            for k in k_values:
                key = f"{method}_{task}_k{k}"
                if key in summary['by_task']:
                    stats = summary['by_task'][key]
                    task_results.append(stats['mean'])
            
            if task_results:
                avg_across_k = np.mean(task_results)
                print(f"  {method:10}: {avg_across_k:.3f} (avg across k-shots)")
    
    # Print results by k-shot
    print("\nRESULTS BY K-SHOT:")
    print("-" * 40)
    for k in k_values:
        print(f"\n{k}-shot:")
        for method in methods:
            key = f"{method}_k{k}"
            if key in summary['by_k_shots']:
                stats = summary['by_k_shots'][key]
                print(f"  {method:10}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
    
    # Print results by task type and k-shot
    print("\nRESULTS BY TASK TYPE AND K-SHOT:")
    print("-" * 50)
    
    task_types = ['classification', 'non_classification']
    
    for task_type in task_types:
        print(f"\n{task_type.upper().replace('_', ' ')}:")
        
        # Create header
        header = f"{'K-Shot':>8}"
        for method in methods:
            header += f"{method.upper():>12}"
        print(header)
        print("-" * len(header))
        
        # Print data for each k-shot
        for k in k_values:
            row = f"{k:>8}"
            for method in methods:
                key = f"{task_type}_{method}_k{k}"
                if key in summary['by_task_type']:
                    stats = summary['by_task_type'][key]
                    row += f"{stats['mean']:12.3f}"
                else:
                    row += f"{'N/A':>12}"
            print(row)
    
    # Print detailed breakdown
    print("\nDETAILED BREAKDOWN (METHOD x TASK x K-SHOT):")
    print("-" * 60)
    
    for task in tasks:
        print(f"\n{task.upper()}:")
        header = f"{'Method':12}"
        for k in k_values:
            header += f"{k:>12}-shot"
        print(header)
        print("-" * len(header))
        
        for method in methods:
            row = f"{method:12}"
            for k in k_values:
                key = f"{method}_{task}_k{k}"
                if key in summary['by_task']:
                    stats = summary['by_task'][key]
                    row += f"{stats['mean']:12.3f}"
                else:
                    row += f"{'N/A':>12}"
            print(row)

if __name__ == "__main__":
    main() 