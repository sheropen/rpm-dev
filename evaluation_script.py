#!/usr/bin/env python3
"""
Reusable evaluation script for Generative Adapter on LongProc tasks
Supports multiple tasks with a modular design
"""

import json
import yaml
import os
import torch
import logging
from typing import Dict, List, Any, Optional, Callable
import re
import sys
import argparse
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime

# Add the generative-adapter root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'generative-adapter'))

# Import generative adapter functions
from fastlora.config import FastLoraConfig
from fastlora.model import FastLoraModelForCausalLM, FastLoraModel, get_peft_model_state_dict, set_peft_model_state_dict, load_pretrained_model
import peft.peft_model as peft_model
import peft.mapping as peft_mapping
from fastlora.eval_utils import fastlora_generate_adaptor, fastlora_conditional_generate
from peft.config import PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Monkey patching (same as inference.py)
peft_model.PEFT_TYPE_TO_MODEL_MAPPING.update({"FASTLORA": FastLoraModel})
peft_mapping.PEFT_TYPE_TO_CONFIG_MAPPING.update({"FASTLORA": FastLoraConfig})
peft_model.get_peft_model_state_dict = get_peft_model_state_dict

# Create a wrapper for set_peft_model_state_dict to handle additional parameters
def patched_set_peft_model_state_dict(model, peft_model_state_dict, adapter_name="default", **kwargs):
    # Filter out unsupported kwargs
    supported_kwargs = {k: v for k, v in kwargs.items() if k in ['ignore_mismatched_sizes']}
    return set_peft_model_state_dict(model, peft_model_state_dict, adapter_name, **supported_kwargs)

peft_model.set_peft_model_state_dict = patched_set_peft_model_state_dict

# Monkey patch PEFT to accept custom task types (same as inference.py)
import peft.config
original_post_init = peft.config.PeftConfig.__post_init__

def patched_post_init(self):
    # Skip task type validation for custom types
    if hasattr(self, 'task_type') and self.task_type.startswith('FAST_LORA_'):
        return
    return original_post_init(self)

peft.config.PeftConfig.__post_init__ = patched_post_init


class TaskEvaluator(ABC):
    """Abstract base class for task-specific evaluators"""
    
    def __init__(self, task_name: str, data_dir: str, prompts_file: str, input_prompt_template_file: str):
        self.task_name = task_name
        self.data_dir = data_dir
        self.prompts_file = prompts_file
        self.input_prompt_template_file = input_prompt_template_file
        
    @abstractmethod
    def load_sample_content(self, sample: Dict[str, Any]) -> str:
        """Load the content for a sample based on task type"""
        pass
    
    @abstractmethod
    def format_input_prompt(self, sample: Dict[str, Any], content: str) -> str:
        """Format the input prompt template with sample data"""
        pass
    
    @abstractmethod
    def evaluate_output(self, generated_output: str, ground_truth: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the generated output against ground truth"""
        pass
    
    def load_prompts(self) -> str:
        """Load the raw prompts from YAML file"""
        with open(self.prompts_file, 'r') as f:
            prompts = yaml.safe_load(f)
        return prompts['USER_PROMPT']
    
    def format_prompts(self, sample: Dict[str, Any]) -> str:
        """Format the prompts with sample data"""
        raw_prompts = self.load_prompts()
        return self._format_prompts_with_sample(raw_prompts, sample)
    
    def _format_prompts_with_sample(self, raw_prompts: str, sample: Dict[str, Any]) -> str:
        """Format prompts with sample data - to be implemented by subclasses"""
        return raw_prompts
    
    def load_input_prompt_template(self) -> str:
        """Load the input prompt template from YAML file"""
        with open(self.input_prompt_template_file, 'r') as f:
            template = yaml.safe_load(f)
        return template['INPUT_PROMPT']


class PseudoToCodeEvaluator(TaskEvaluator):
    """Evaluator for pseudo_to_code task"""
    
    def load_sample_content(self, sample: Dict[str, Any]) -> str:
        """Load pseudocode from sample"""
        pseudocode_lines = sample.get('pseudocode_lines', [])
        return '\n'.join(pseudocode_lines)
    
    def _format_prompts_with_sample(self, raw_prompts: str, sample: Dict[str, Any]) -> str:
        """Format prompts with sample data for pseudo_to_code task"""
        # For pseudo_to_code, the prompts don't need sample-specific formatting
        # The prompts are generic instructions for C++ code generation
        return raw_prompts
    
    def format_input_prompt(self, sample: Dict[str, Any], content: str) -> str:
        """Format input prompt with pseudocode"""
        template = self.load_input_prompt_template()
        return template.format(pseudocode=content)
    
    def evaluate_output(self, generated_output: str, ground_truth: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate C++ code generation"""
        # Extract C++ code from generated output
        cpp_match = re.search(r'```cpp\s*\n(.*?)\n```', generated_output, re.DOTALL)
        if cpp_match:
            extracted_code = cpp_match.group(1).strip()
        else:
            extracted_code = generated_output.strip()
        
        # Get ground truth code lines
        gt_code_lines = sample.get('code_lines', [])
        gt_code = '\n'.join(gt_code_lines)
        
        # Simple evaluation metrics
        gt_lines = gt_code.strip().split('\n')
        gen_lines = extracted_code.strip().split('\n')
        
        # Count lines
        gt_line_count = len(gt_lines)
        gen_line_count = len(gen_lines)
        
        # Check if main function is present
        has_main = 'int main()' in extracted_code or 'int main(' in extracted_code
        
        # Check if includes are present (should not be included according to prompt)
        has_includes = '#include' in extracted_code
        
        return {
            'gt_line_count': gt_line_count,
            'gen_line_count': gen_line_count,
            'has_main': has_main,
            'has_includes': has_includes,
            'extracted_code': extracted_code,
            'gt_code': gt_code
        }


class HtmlToTsvEvaluator(TaskEvaluator):
    """Evaluator for html_to_tsv task"""
    
    def load_sample_content(self, sample: Dict[str, Any]) -> str:
        """Load HTML content from file"""
        html_path = sample.get('html_path', '')
        full_path = os.path.join(self.data_dir, html_path)
        
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _format_prompts_with_sample(self, raw_prompts: str, sample: Dict[str, Any]) -> str:
        """Format prompts with sample data for html_to_tsv task"""
        # Extract fields from sample
        task_topic = sample.get('task_topic', '')
        task_description = sample.get('task_description', '')
        tsv_header = sample.get('tsv_header', '')
        filtering_instruction = sample.get('filtering_instruction', '')
        
        return raw_prompts.format(
            task_topic=task_topic,
            task_description=task_description,
            tsv_header=tsv_header,
            filtering_instruction=filtering_instruction
        )
    
    def format_input_prompt(self, sample: Dict[str, Any], content: str) -> str:
        """Format input prompt with HTML content"""
        template = self.load_input_prompt_template()
        return template.format(html_str=content)
        
    def evaluate_output(self, generated_output: str, ground_truth: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate TSV generation"""
        # Extract TSV content if it's wrapped in code blocks
        tsv_match = re.search(r'```tsv\s*\n(.*?)\n```', generated_output, re.DOTALL)
        if tsv_match:
            extracted_tsv = tsv_match.group(1).strip()
        else:
            # If no code block, try to extract TSV-like content
            lines = generated_output.strip().split('\n')
            tsv_lines = []
            for line in lines:
                if '\t' in line:  # Contains tab separator
                    tsv_lines.append(line)
            extracted_tsv = '\n'.join(tsv_lines) if tsv_lines else generated_output.strip()
        
        # Get ground truth
        gt_lines = ground_truth.strip().split('\n')
        gen_lines = extracted_tsv.strip().split('\n')
        
        # Count rows (excluding header)
        gt_row_count = len(gt_lines) - 1 if len(gt_lines) > 1 else 0
        gen_row_count = len(gen_lines) - 1 if len(gen_lines) > 1 else 0
        
        # Check if headers match
        gt_header = gt_lines[0] if gt_lines else ""
        gen_header = gen_lines[0] if gen_lines else ""
        header_match = gt_header == gen_header
        
        return {
            'gt_row_count': gt_row_count,
            'gen_row_count': gen_row_count,
            'header_match': header_match,
            'extracted_tsv': extracted_tsv,
            'gt_tsv': ground_truth
        }


class CountdownEvaluator(TaskEvaluator):
    """Evaluator for countdown task"""
    
    def load_sample_content(self, sample: Dict[str, Any]) -> str:
        """Load demonstration from sample"""
        return sample.get('demonstration', '')
    
    def _format_prompts_with_sample(self, raw_prompts: str, sample: Dict[str, Any]) -> str:
        """Format prompts with sample data for countdown task"""
        # For countdown, the prompts don't need sample-specific formatting
        # The prompts are generic instructions for countdown problem solving
        return raw_prompts
    
    def format_input_prompt(self, sample: Dict[str, Any], content: str) -> str:
        """Format input prompt with demonstration"""
        template = self.load_input_prompt_template()
        return template.format(demonstration=content)
    
    def evaluate_output(self, generated_output: str, ground_truth: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate countdown solution"""
        # Look for <Solution> tags
        solution_match = re.search(r'<Solution>(.*?)</Solution>', generated_output, re.DOTALL)
        if solution_match:
            extracted_solution = solution_match.group(1).strip()
        else:
            extracted_solution = generated_output.strip()
        
        # Get ground truth
        gt_lines = ground_truth.strip().split('\n')
        gen_lines = extracted_solution.strip().split('\n')
        
        # Count lines
        gt_line_count = len(gt_lines)
        gen_line_count = len(gen_lines)
        
        return {
            'gt_line_count': gt_line_count,
            'gen_line_count': gen_line_count,
            'extracted_solution': extracted_solution,
            'gt_solution': ground_truth
        }


class PathTraversalEvaluator(TaskEvaluator):
    """Evaluator for path_traversal task"""
    
    def load_sample_content(self, sample: Dict[str, Any]) -> str:
        """Load city context from sample"""
        return sample.get('context_nl', '')
    
    def _format_prompts_with_sample(self, raw_prompts: str, sample: Dict[str, Any]) -> str:
        """Format prompts with sample data for path_traversal task"""
        # For path_traversal, the prompts don't need sample-specific formatting
        # The prompts are generic instructions for path finding
        return raw_prompts
    
    def format_input_prompt(self, sample: Dict[str, Any], content: str) -> str:
        """Format input prompt with city context and route query"""
        template = self.load_input_prompt_template()
        
        # Extract source and destination cities from question_nl
        question_nl = sample.get('question_nl', '')
        # # Parse "Please give a route from X to Y." format
        # import re
        # route_match = re.search(r'from (\w+(?:\s+\w+)*) to (\w+(?:\s+\w+)*)', question_nl)
        # if route_match:
        #     src_city = route_match.group(1).strip()
        #     dst_city = route_match.group(2).strip()
        # else:
        #     src_city = "unknown"
        #     dst_city = "unknown"
        
        return template.format(
            city_context=content,
            question_nl=question_nl
        )
    
    def evaluate_output(self, generated_output: str, ground_truth: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate path traversal solution"""
        # Extract route from generated output
        route_match = re.search(r'<Route>(.*?)</Route>', generated_output, re.DOTALL)
        if route_match:
            extracted_route = route_match.group(1).strip()
        else:
            extracted_route = generated_output.strip()
        
        # Get ground truth route
        gt_route = ground_truth.strip()
        
        # Count route steps
        gt_steps = [line.strip() for line in gt_route.split('\n') if line.strip()]
        gen_steps = [line.strip() for line in extracted_route.split('\n') if line.strip()]
        
        # Count steps
        gt_step_count = len(gt_steps)
        gen_step_count = len(gen_steps)
        
        # Check if route format is correct (starts with "From" and contains transit methods)
        has_correct_format = all(
            step.startswith('From ') and any(method in step for method in ['bus', 'train', 'plane', 'ferry'])
            for step in gen_steps
        )
        
        return {
            'gt_step_count': gt_step_count,
            'gen_step_count': gen_step_count,
            'has_correct_format': has_correct_format,
            'extracted_route': extracted_route,
            'gt_route': gt_route
        }


class GenerativeAdapterEvaluator:
    """Main evaluator class that handles model loading and evaluation"""
    
    def __init__(self, model_name_or_path: str = "generative-adaptor/Generative-Adapter-Mistral-7B-Instruct-v0.2"):
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.tokenizer = None
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        try:
            device = 'cuda'
            torch_dtype = torch.bfloat16
            attn_implementation = 'sdpa'

            logger.info(f"Loading PEFT config from {self.model_name_or_path}")
            peft_config = PeftConfig.from_pretrained(self.model_name_or_path)

            base_model_path = peft_config.base_model_name_or_path
            assert base_model_path is not None, "base_model_name_or_path should not be None"
            logger.info(f"Base model path: {base_model_path}")

            logger.info("Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                low_cpu_mem_usage=False,
            )
            logger.info("Base model loaded successfully")

            logger.info("Loading FastLora model...")
            self.model = FastLoraModelForCausalLM.from_pretrained(
                base_model,
                self.model_name_or_path,
                adapter_name='default',
                is_trainable=False,
                config=peft_config,
            ).cuda()
            logger.info("FastLora model loaded successfully")

            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            logger.info("Tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def generate_lora_weights(self, prompt_prefix: str, merge_strategy: str = 'concat', window_size: int = 1024) -> List[torch.Tensor]:
        """Generate LoRA weights from prompt prefix"""
        logger.info("Generating adapter weights from prompt template...")
        lora_weights = fastlora_generate_adaptor(
            self.model, 
            self.tokenizer, 
            prompt_prefix, 
            merge_strategy=merge_strategy, 
            max_window_size=window_size,
        )
        logger.info(f"Generated {len(lora_weights)} LoRA weights")
        return lora_weights
    
    def generate_output(self, input_prompt: str, lora_weights: List[torch.Tensor], max_new_tokens: int = 1000) -> str:
        """Generate output using the model with LoRA weights"""
        output_text = fastlora_conditional_generate(
            self.model, 
            self.tokenizer, 
            input_text=input_prompt, 
            use_chat=True,
            mode="weights", 
            lora_weights=lora_weights, 
            max_new_tokens=max_new_tokens,
            stop=[]
        )
        return output_text
    
    def evaluate_task(self, task_evaluator: TaskEvaluator, dataset_file: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate a specific task"""
        logger.info(f"Evaluating task: {task_evaluator.task_name}")
        
        # Load dataset
        logger.info(f"Loading dataset from {dataset_file}")
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        
        if max_samples:
            dataset = dataset[:max_samples]
            logger.info(f"Testing on {len(dataset)} samples")
        
        # Check if the task needs sample-specific prompt formatting
        needs_sample_specific_prompts = task_evaluator.task_name in ["html_to_tsv"]
        
        if needs_sample_specific_prompts:
            logger.info(f"Task {task_evaluator.task_name} requires sample-specific prompts - will regenerate LoRA weights for each sample")
            logger.info("This is because the prompt template contains placeholders that need to be filled with sample-specific data")
            lora_weights = None  # Will be generated per sample
        else:
            # Use the first sample to format prompts for LoRA weight generation
            # This ensures the prompts are properly formatted with sample-specific information
            first_sample = dataset[0] if dataset else {}
            logger.info(f"Task {task_evaluator.task_name} uses generic prompts - formatting with first sample: {first_sample.get('task_id', first_sample.get('problem_id', 'unknown'))}")
            formatted_prompts = task_evaluator.format_prompts(first_sample)
            logger.info(f"Formatted prompts length: {len(formatted_prompts)} characters")
            
            # Generate LoRA weights using formatted prompts
            lora_weights = self.generate_lora_weights(formatted_prompts)
        
        # Evaluate each sample
        results = []
        for i, sample in enumerate(dataset):
            logger.info(f"Processing sample {i+1}/{len(dataset)}")
            
            try:
                # Load sample content
                content = task_evaluator.load_sample_content(sample)
                
                # Generate LoRA weights if needed for this sample
                if needs_sample_specific_prompts:
                    logger.info(f"Generating LoRA weights for sample {i+1}")
                    formatted_prompts = task_evaluator.format_prompts(sample)
                    
                    # xy edited
                    print("formatted_prompts", formatted_prompts)
                    
                    sample_lora_weights = self.generate_lora_weights(formatted_prompts)
                else:
                    sample_lora_weights = lora_weights
                    # For non-sample-specific tasks, get the formatted prompts used for LoRA generation
                    formatted_prompts = task_evaluator.format_prompts(sample)
                
                # Format input prompt
                input_prompt = task_evaluator.format_input_prompt(sample, content)
                
                # Generate output
                generated_output = self.generate_output(input_prompt, sample_lora_weights)
                
                # Get ground truth
                ground_truth = sample.get('gt', '')
                
                # Evaluate output
                evaluation = task_evaluator.evaluate_output(generated_output, ground_truth, sample)
                
                result = {
                    'task_id': sample.get('task_id', sample.get('problem_id', f'sample_{i}')),
                    'ground_truth': ground_truth,
                    'generated_output': generated_output,
                    'task_prompt': formatted_prompts,  # The formatted prompts used for LoRA generation
                    'input_prompt': input_prompt,      # The formatted input prompt used for generation
                    'success': True,
                    **evaluation
                }
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {str(e)}")
                result = {
                    'task_id': sample.get('task_id', sample.get('problem_id', f'sample_{i}')),
                    'error': str(e),
                    'success': False,
                    'task_prompt': formatted_prompts if 'formatted_prompts' in locals() else None,
                    'input_prompt': input_prompt if 'input_prompt' in locals() else None
                }
            
            results.append(result)
        
        # Calculate summary statistics
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        summary = {
            'task_name': task_evaluator.task_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_name': self.model_name_or_path,
            'total_samples': len(dataset),
            'successful_samples': len(successful_results),
            'failed_samples': len(failed_results),
            'success_rate': len(successful_results) / len(dataset) if dataset else 0,
            'results': results
        }
        
        # Add task-specific metrics
        if successful_results:
            if 'header_match' in successful_results[0]:
                summary['header_match_rate'] = sum(1 for r in successful_results if r['header_match']) / len(successful_results)
            if 'has_main' in successful_results[0]:
                summary['main_function_rate'] = sum(1 for r in successful_results if r['has_main']) / len(successful_results)
                summary['includes_rate'] = sum(1 for r in successful_results if r['has_includes']) / len(successful_results)
        
        return summary


def get_task_config(task_type: str) -> Dict[str, str]:
    """Get configuration for different tasks"""
    task_configs = {
        "html_to_tsv": {
            "dataset_file": "LongProc/data/html_to_tsv/html_to_tsv_0.5k.json",
            "prompts_file": "prompts/html_to_tsv/prompts.yaml",
            "input_prompt_template_file": "prompts/html_to_tsv/input_prompt_template.yml",
            "data_dir": "LongProc/data/html_to_tsv"
        },
        "pseudo_to_code": {
            "dataset_file": "LongProc/data/pseudo_to_code/pseudo_to_code_0.5k.json",
            "prompts_file": "prompts/pseudo_to_code/prompts.yaml",
            "input_prompt_template_file": "prompts/pseudo_to_code/input_prompt_template.yml",
            "data_dir": "LongProc/data/pseudo_to_code"
        },
        "countdown": {
            "dataset_file": "LongProc/data/countdown/countdown_0.5k.json",
            "prompts_file": "prompts/countdown/prompts.yaml",
            "input_prompt_template_file": "prompts/countdown/input_prompt_template.yml",
            "data_dir": "LongProc/data/countdown"
        },
        "path_traversal": {
            "dataset_file": "LongProc/data/path_traversal/path_traversal_0.5k.json",
            "prompts_file": "prompts/path_traversal/prompts.yaml",
            "input_prompt_template_file": "prompts/path_traversal/input_prompt_template.yaml",
            "data_dir": "LongProc/data/path_traversal"
        }
    }
    
    if task_type not in task_configs:
        raise ValueError(f"Unsupported task type: {task_type}. Supported tasks: {list(task_configs.keys())}")
    
    return task_configs[task_type]


def create_task_evaluator(task_type: str, config: Dict[str, str]) -> TaskEvaluator:
    """Create task-specific evaluator"""
    if task_type == "html_to_tsv":
        return HtmlToTsvEvaluator(
            task_type, 
            config['data_dir'], 
            config['prompts_file'], 
            config['input_prompt_template_file']
        )
    elif task_type == "pseudo_to_code":
        return PseudoToCodeEvaluator(
            task_type, 
            config['data_dir'], 
            config['prompts_file'], 
            config['input_prompt_template_file']
        )
    elif task_type == "countdown":
        return CountdownEvaluator(
            task_type, 
            config['data_dir'], 
            config['prompts_file'], 
            config['input_prompt_template_file']
        )
    elif task_type == "path_traversal":
        return PathTraversalEvaluator(
            task_type, 
            config['data_dir'], 
            config['prompts_file'], 
            config['input_prompt_template_file']
        )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def create_evaluation_summary(output_dir: Path) -> Dict[str, Any]:
    """Create a summary of all evaluation results in the directory"""
    summary = {
        'summary_timestamp': datetime.now().isoformat(),
        'total_evaluations': 0,
        'evaluations': []
    }
    
    # Find all JSON files in the output directory
    json_files = list(output_dir.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract key information
            eval_info = {
                'filename': json_file.name,
                'task_name': data.get('task_name', 'unknown'),
                'evaluation_timestamp': data.get('evaluation_timestamp', 'unknown'),
                'model_name': data.get('model_name', 'unknown'),
                'total_samples': data.get('total_samples', 0),
                'successful_samples': data.get('successful_samples', 0),
                'success_rate': data.get('success_rate', 0.0)
            }
            
            # Add task-specific metrics
            if 'header_match_rate' in data:
                eval_info['header_match_rate'] = data['header_match_rate']
            if 'main_function_rate' in data:
                eval_info['main_function_rate'] = data['main_function_rate']
                eval_info['includes_rate'] = data.get('includes_rate', 0.0)
            
            summary['evaluations'].append(eval_info)
            summary['total_evaluations'] += 1
            
        except Exception as e:
            logger.warning(f"Could not read {json_file}: {e}")
    
    # Sort evaluations by timestamp (newest first)
    summary['evaluations'].sort(key=lambda x: x['evaluation_timestamp'], reverse=True)
    
    return summary


def main():
    """Main function to run the evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate Generative Adapter on LongProc tasks')
    parser.add_argument('--task', type=str, 
                       default="pseudo_to_code",
                       choices=["html_to_tsv", "pseudo_to_code", "countdown", "path_traversal"],
                       help='Task type to evaluate')
    parser.add_argument('--dataset', type=str, 
                       help='Path to dataset file (overrides task default)')
    parser.add_argument('--max-samples', type=int, default=3,
                       help='Maximum number of samples to test (0 for all)')
    parser.add_argument('--output-dir', type=str, default="evaluation_results",
                       help='Output directory for results')
    parser.add_argument('--model', type=str, 
                       default="generative-adaptor/Generative-Adapter-Mistral-7B-Instruct-v0.2",
                       help='Model name or path')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only generate summary of existing results, skip evaluation')
    
    args = parser.parse_args()
    
    # Get task configuration
    task_config = get_task_config(args.task)
    
    # Override with command line arguments
    if args.dataset:
        task_config['dataset_file'] = args.dataset
    
    max_samples = args.max_samples if args.max_samples > 0 else None
    
    # Create output directory and filename
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create descriptive filename
    sample_info = f"{max_samples}samples" if max_samples else "allsamples"
    output_filename = f"{args.task}_{sample_info}_{timestamp}.json"
    output_path = output_dir / output_filename
    
    logger.info(f"Evaluating task: {args.task}")
    logger.info(f"Dataset file: {task_config['dataset_file']}")
    logger.info(f"Prompts file: {task_config['prompts_file']}")
    logger.info(f"Input prompt template file: {task_config['input_prompt_template_file']}")
    logger.info(f"Output will be saved to: {output_path}")
    
    # Handle summary-only mode
    if args.summary_only:
        logger.info("Generating summary of existing evaluation results...")
        summary_data = create_evaluation_summary(output_dir)
        summary_path = output_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")
        return
    
    try:
        # Create evaluator
        evaluator = GenerativeAdapterEvaluator(args.model)
        
        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        evaluator.load_model_and_tokenizer()
        logger.info("Model and tokenizer loaded successfully!")
        
        # Create task-specific evaluator
        task_evaluator = create_task_evaluator(args.task, task_config)
        
        # Run evaluation
        summary = evaluator.evaluate_task(task_evaluator, task_config['dataset_file'], max_samples)
        
        # Print summary
        logger.info("Evaluation completed!")
        logger.info(f"Summary: {summary['successful_samples']}/{summary['total_samples']} successful, "
                   f"Success rate: {summary['success_rate']:.2%}")
        
        if 'header_match_rate' in summary:
            logger.info(f"Header match rate: {summary['header_match_rate']:.2%}")
        if 'main_function_rate' in summary:
            logger.info(f"Main function rate: {summary['main_function_rate']:.2%}")
            logger.info(f"Includes rate: {summary['includes_rate']:.2%}")
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        
        # Create and save evaluation summary
        logger.info("Creating evaluation summary...")
        summary_data = create_evaluation_summary(output_dir)
        summary_path = output_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")
        
        # Print detailed results for successful samples
        print("\n=== Detailed Results ===")
        for result in summary['results']:
            if result['success']:
                print(f"\nTask: {result['task_id']}")
                if 'gt_row_count' in result:
                    print(f"Ground Truth Rows: {result['gt_row_count']}")
                    print(f"Generated Rows: {result['gen_row_count']}")
                    print(f"Header Match: {result['header_match']}")
                elif 'gt_line_count' in result:
                    print(f"Ground Truth Lines: {result['gt_line_count']}")
                    print(f"Generated Lines: {result['gen_line_count']}")
                    if 'has_main' in result:
                        print(f"Has Main Function: {result['has_main']}")
                        print(f"Has Includes: {result['has_includes']}")
                
                print(f"Generated Output Preview:")
                output_preview = result['generated_output'][:500] + "..." if len(result['generated_output']) > 500 else result['generated_output']
                print(output_preview)
                
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
