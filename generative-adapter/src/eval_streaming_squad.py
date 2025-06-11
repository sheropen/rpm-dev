#!/usr/bin/env python3
"""
Evaluation script for StreamingQA and SQuAD datasets using FastLora models.
Supports both context-aware and context-free QA evaluation.
"""

import json
import logging
import collections
import re
import string
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Union
import argparse
import traceback
import time

# FastLora imports
from fastlora.eval_utils import fastlora_generate_adaptor, fastlora_conditional_generate
from fastlora.model import FastLoraModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastlora.config import FastLoraConfig
from fastlora.model import FastLoraModel, get_peft_model_state_dict, set_peft_model_state_dict
import peft.peft_model as peft_model
import peft.mapping as peft_mapping

## monkey patching
peft_model.PEFT_TYPE_TO_MODEL_MAPPING.update({"FASTLORA": FastLoraModel})
peft_mapping.PEFT_TYPE_TO_CONFIG_MAPPING.update({"FASTLORA": FastLoraConfig})
peft_model.get_peft_model_state_dict = get_peft_model_state_dict
peft_model.set_peft_model_state_dict = set_peft_model_state_dict

from peft.config import PeftConfig

# Set up basic logging (will be enhanced in main())
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging(output_dir: Path):
    """Set up logging to both console and file."""
    # Create log file path
    log_file = output_dir / "evaluation.log"
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Set logger level
    logger.setLevel(logging.INFO)
    
    logger.info(f"Logging setup complete. Log file: {log_file}")
    return log_file

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def load_wmt_docs_index(wmt_docs_path: str) -> Dict[str, str]:
    """Load WMT documents and create index mapping sorting_key to text."""
    logger.info(f"Loading WMT documents index from {wmt_docs_path}")
    
    if not Path(wmt_docs_path).exists():
        logger.warning(f"WMT docs file not found: {wmt_docs_path}")
        return {}
    
    docs_index = {}
    with open(wmt_docs_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                doc = json.loads(line.strip())
                sorting_key = doc['sorting_key']
                text = doc['text']
                docs_index[sorting_key] = text
                
                if line_num % 10000 == 0:
                    logger.info(f"Loaded {line_num} WMT documents into index")
                    
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error parsing WMT doc at line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(docs_index)} WMT documents into index")
    return docs_index

def load_streamingqa_dataset(file_path: str, wmt_docs_path: str = None) -> List[Dict[str, Any]]:
    """Load StreamingQA dataset from JSONL file with optional WMT context mapping."""
    logger.info(f"Loading StreamingQA dataset from {file_path}")
    
    # Load WMT docs index if path provided
    wmt_docs_index = {}
    if wmt_docs_path:
        wmt_docs_index = load_wmt_docs_index(wmt_docs_path)
        logger.info(f"Will map evidence_id to context using {len(wmt_docs_index)} WMT documents")
    
    data = []
    missing_evidence_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            # Combine main answers and additional answers
            all_answers = item['answers'] + item.get('answers_additional', [])
            # Remove duplicates while preserving order
            seen = set()
            unique_answers = []
            for ans in all_answers:
                if ans not in seen:
                    unique_answers.append(ans)
                    seen.add(ans)
            
            # Get context from WMT docs if available
            context = ""
            evidence_id = item.get('evidence_id', '')
            if evidence_id and wmt_docs_index:
                context = wmt_docs_index.get(evidence_id, "")
                if not context:
                    missing_evidence_count += 1
                    logger.debug(f"No context found for evidence_id: {evidence_id}")
            
            data.append({
                'id': item['qa_id'],
                'question': item['question'],
                'answers': unique_answers,
                'context': context,
                'evidence_id': evidence_id,
                'dataset': 'streamingqa'
            })
    
    logger.info(f"Loaded {len(data)} StreamingQA examples")
    if wmt_docs_path:
        found_context_count = len(data) - missing_evidence_count
        logger.info(f"Found context for {found_context_count}/{len(data)} examples ({found_context_count/len(data)*100:.1f}%)")
        if missing_evidence_count > 0:
            logger.warning(f"Missing context for {missing_evidence_count} examples")
    
    return data

def load_squad_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load SQuAD dataset from JSON file."""
    logger.info(f"Loading SQuAD dataset from {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)
    
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                # Extract all possible answers
                answers = []
                if qa.get('answers'):
                    answers.extend([ans['text'] for ans in qa['answers']])
                if qa.get('plausible_answers'):  # SQuAD 2.0
                    answers.extend([ans['text'] for ans in qa['plausible_answers']])
                
                # If no answers (unanswerable in SQuAD 2.0), use empty string
                if not answers:
                    answers = ['']
                
                data.append({
                    'id': qa['id'],
                    'question': qa['question'],
                    'context': context,
                    'answers': answers,
                    'dataset': 'squad'
                })
    
    logger.info(f"Loaded {len(data)} SQuAD examples")
    return data

def generate_answer_fastlora(
    model, tokenizer, question: str, context: str = None, 
    use_context: bool = True, max_new_tokens: int = 50
) -> str:
    """Generate answer using FastLora model."""
    try:
        # Step 1: Generate LoRA weights from context
        logger.debug("Generating LoRA weights from context")
        lora_weights = fastlora_generate_adaptor(
            model=model,
            tokenizer=tokenizer,
            context_text=context,
            merge_strategy="sequential",
            max_window_size=1024
        )
        
        # Step 2: Generate answer conditioned on context
        answer = fastlora_conditional_generate(
            model=model,
            tokenizer=tokenizer,
            input_text=question,
            mode="weights",
            lora_weights=lora_weights,
            use_chat=True,
            max_new_tokens=max_new_tokens,
            stop=["\n", ".", "?", "!"]
        )
        
        logger.debug(f"question: {question}")
        logger.debug(f"answer: {answer}")

        return answer.strip()
    
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        logger.error(traceback.format_exc())
        return ""

def generate_answer_icl(
    model, tokenizer, question: str, context: str = None, 
    use_context: bool = True, max_new_tokens: int = 50
) -> str:
    """Generate answer using in-context learning with base model."""
    try:
        if use_context and context:
            # Create ICL prompt with context
            prompt = f"""{context}  
## Instruction: Answer the question based on the context above. Respond with a short phrase only. Keep the answer short and concise, without any explanation or additional words  Question: {question} 
Answer:"""
        else:
            # Direct question without context
            prompt = f"""## Instruction: Answer the question. Respond with a short phrase only. Keep the answer short and concise, without any explanation or additional words  Question: {question} 
Answer:"""
        
        logger.debug(f"ICL prompt: {prompt}")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode only the generated part (remove input prompt)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up answer - stop at first newline, period, etc.
        for stop_token in ["\n", ".", "?", "!"]:
            if stop_token in answer:
                answer = answer.split(stop_token)[0]
                break
        
        logger.debug(f"question: {question}")
        logger.debug(f"answer: {answer}")
        
        return answer.strip()
    
    except Exception as e:
        logger.error(f"Error generating ICL answer: {e}")
        logger.error(traceback.format_exc())
        return ""

def evaluate_dataset(
    model, tokenizer, dataset: List[Dict[str, Any]], 
    use_context: bool = True, max_examples: int = None, mode: str = 'ga'
) -> List[Dict[str, Any]]:
    """Evaluate model on dataset and return results with scores."""
    
    if max_examples:
        dataset = dataset[:max_examples]
        logger.info(f"Evaluating on first {max_examples} examples")
    
    results = []
    
    for i, item in enumerate(tqdm(dataset, desc="Evaluating")):
        # logger.info(f"item: {item}")
        # import pdb; pdb.set_trace() 
        
        # Generate prediction
        context = item.get('context', '')
        
        if mode == 'ga':
            prediction = generate_answer_fastlora(
                model=model,
                tokenizer=tokenizer,
                question=item['question'],
                context=context,
                use_context=use_context and bool(context)
            )
        elif mode == 'icl':
            prediction = generate_answer_icl(
                model=model,
                tokenizer=tokenizer,
                question=item['question'],
                context=context,
                use_context=use_context and bool(context)
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Compute scores against all gold answers
        gold_answers = [str(ans) for ans in item['answers']]
        exact_scores = [compute_exact(ans, prediction) for ans in gold_answers]
        f1_scores = [compute_f1(ans, prediction) for ans in gold_answers]
        
        # Take maximum score (SQuAD evaluation standard)
        max_exact = max(exact_scores) if exact_scores else 0
        max_f1 = max(f1_scores) if f1_scores else 0
        
        result = {
            'id': item['id'],
            'dataset': item['dataset'],
            'question': item['question'],
            'context': context,
            'gold_answers': gold_answers,
            'prediction': prediction,
            'scores': {
                'exact_match': max_exact,
                'f1': max_f1
            }
        }
        
        results.append(result)
        
        # Log progress
        if (i + 1) % 50 == 0:
            avg_f1 = sum(r['scores']['f1'] for r in results[-50:]) / 50
            avg_em = sum(r['scores']['exact_match'] for r in results[-50:]) / 50
            logger.info(f"Progress {i+1}/{len(dataset)} - Last 50: F1={avg_f1:.3f}, EM={avg_em:.3f}")

        # TODO: for testing only, remove this    
        # if i == 10:
        #     break
    
    return results

def compute_final_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute final aggregated metrics."""
    if not results:
        return {}
    
    # Overall metrics
    total_f1 = sum(r['scores']['f1'] for r in results)
    total_em = sum(r['scores']['exact_match'] for r in results)
    total_count = len(results)
    
    metrics = {
        'total_examples': total_count,
        'average_f1': total_f1 / total_count,
        'average_exact_match': total_em / total_count
    }
    
    # Per-dataset metrics
    datasets = set(r['dataset'] for r in results)
    for dataset in datasets:
        dataset_results = [r for r in results if r['dataset'] == dataset]
        if dataset_results:
            dataset_f1 = sum(r['scores']['f1'] for r in dataset_results) / len(dataset_results)
            dataset_em = sum(r['scores']['exact_match'] for r in dataset_results) / len(dataset_results)
            metrics[f'{dataset}_f1'] = dataset_f1
            metrics[f'{dataset}_exact_match'] = dataset_em
            metrics[f'{dataset}_count'] = len(dataset_results)
    
    return metrics

def load_fastlora_model(model_name_or_path: str, device: str = 'cuda', torch_dtype = torch.bfloat16, attn_implementation: str = 'sdpa'):
    """Load FastLora model using the same approach as inference.ipynb"""
    logger.info(f"Loading FastLora model from {model_name_or_path}")
    
    # Load the PEFT configuration from the given pretrained model directory
    logger.info("Loading PEFT configuration...")
    peft_config = PeftConfig.from_pretrained(model_name_or_path)
    
    # Get the base model path from the configuration
    base_model_path = peft_config.base_model_name_or_path
    assert base_model_path is not None, "base_model_name_or_path should not be None"
    logger.info(f"Base model path: {base_model_path}")
    logger.info(f"attn_implementation: {attn_implementation}")
    
    # Load the base causal language model from the retrieved base model path
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    
    # Load the FastLora model for causal language modeling
    logger.info("Loading FastLora model...")
    model = FastLoraModelForCausalLM.from_pretrained(
        base_model,
        model_name_or_path,
        adapter_name='default',
        is_trainable=False,
        config=peft_config,
    ).to(device)
    
    # Load the tokenizer from the pretrained model directory
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    logger.info("Model loading completed successfully!")
    return model, tokenizer

def load_base_model_for_icl(model_name_or_path: str, device: str = 'cuda', torch_dtype = torch.bfloat16, attn_implementation: str = 'sdpa'):
    """Load base model for in-context learning (ICL mode)"""
    logger.info(f"Loading base model for ICL from {model_name_or_path}")
    
    # For ICL, we first try to load PEFT config to get base model path, 
    # but if that fails, assume model_name_or_path is already the base model
    try:
        # Try to load PEFT config to get base model path
        logger.info("Trying to load PEFT configuration to get base model path...")
        peft_config = PeftConfig.from_pretrained(model_name_or_path)
        base_model_path = peft_config.base_model_name_or_path
        logger.info(f"Found base model path from PEFT config: {base_model_path}")
    except Exception as e:
        # If no PEFT config, assume the provided path is already the base model
        logger.info(f"No PEFT config found, using provided path as base model: {model_name_or_path}")
        base_model_path = model_name_or_path
    
    logger.info(f"Base model path: {base_model_path}")
    logger.info(f"attn_implementation: {attn_implementation}")
    
    # Load the base causal language model
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    ).to(device)
    
    # Load the tokenizer - try from adapter path first, then base model path
    logger.info("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    except Exception as e:
        logger.info(f"Failed to load tokenizer from {model_name_or_path}, trying base model path...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Base model loading completed successfully!")
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description='Evaluate FastLora model on StreamingQA and SQuAD datasets')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the FastLora model')
    parser.add_argument('--streamingqa_path', type=str, help='Path to StreamingQA JSONL file')
    parser.add_argument('--squad_path', type=str, help='Path to SQuAD JSON file')
    parser.add_argument('--wmt_docs_path', type=str, help='Path to processed WMT docs JSONL file for StreamingQA context mapping (output from extract.py)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--max_examples', type=int, default=None, help='Maximum number of examples to evaluate')
    parser.add_argument('--use_context', action='store_true', default=True, help='Use context for generation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run model on')
    parser.add_argument('--torch_dtype', type=str, default='bfloat16', help='Torch dtype for model')
    parser.add_argument('--attn_implementation', type=str, default='sdpa', help='Attention implementation')
    parser.add_argument('--mode', type=str, choices=['ga', 'icl'], default='ga', help='Evaluation mode: ga (generative adapter/FastLora) or icl (in-context learning)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to both console and file
    log_file = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("FASTLORA MODEL EVALUATION STARTED")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {log_file}")
    
    # Log all arguments
    logger.info("Evaluation arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Log mode-specific information
    if args.mode == 'ga':
        logger.info("Using Generative Adapter (FastLora) mode")
    elif args.mode == 'icl':
        logger.info("Using In-Context Learning (ICL) mode")
    
    # Start timing
    start_time = time.time()
    
    # Convert torch dtype string to actual dtype
    if args.torch_dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif args.torch_dtype == 'float16':
        torch_dtype = torch.float16
    elif args.torch_dtype == 'float32':
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16
        logger.warning(f"Unknown torch_dtype {args.torch_dtype}, using bfloat16")
    
    # Load model based on mode
    logger.info(f"Loading model from {args.model_path} in {args.mode} mode")
    if args.mode == 'ga':
        model, tokenizer = load_fastlora_model(
            model_name_or_path=args.model_path,
            device=args.device,
            torch_dtype=torch_dtype,
            attn_implementation=args.attn_implementation
        )
    elif args.mode == 'icl':
        model, tokenizer = load_base_model_for_icl(
            model_name_or_path=args.model_path,
            device=args.device,
            torch_dtype=torch_dtype,
            attn_implementation=args.attn_implementation
        )
    
    all_results = []
    
    # Evaluate StreamingQA
    if args.streamingqa_path:
        logger.info("=" * 50)
        logger.info("Evaluating StreamingQA")
        logger.info("=" * 50)
        
        streamingqa_data = load_streamingqa_dataset(
            file_path=args.streamingqa_path,
            wmt_docs_path=args.wmt_docs_path
        )
        
        # Check if we have context for StreamingQA
        has_context = any(item.get('context', '') for item in streamingqa_data)
        use_context_for_streamingqa = args.use_context and has_context
        
        if has_context:
            logger.info(f"StreamingQA evaluation will use context (use_context={use_context_for_streamingqa})")
        else:
            logger.info("StreamingQA evaluation will run without context (no WMT docs provided or no context found)")
        
        streamingqa_results = evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset=streamingqa_data,
            use_context=use_context_for_streamingqa,
            max_examples=args.max_examples,
            mode=args.mode
        )
        all_results.extend(streamingqa_results)
        
        # Save StreamingQA results
        streamingqa_output = output_dir / "streamingqa_results.json"
        with open(streamingqa_output, 'w', encoding='utf-8') as f:
            json.dump(streamingqa_results, f, indent=2, ensure_ascii=False)
        logger.info(f"StreamingQA results saved to {streamingqa_output}")
    
    # Evaluate SQuAD
    if args.squad_path:
        logger.info("=" * 50)
        logger.info("Evaluating SQuAD")
        logger.info("=" * 50)
        
        squad_data = load_squad_dataset(args.squad_path)
        squad_results = evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset=squad_data,
            use_context=args.use_context,
            max_examples=args.max_examples,
            mode=args.mode
        )
        all_results.extend(squad_results)
        
        # Save SQuAD results
        squad_output = output_dir / "squad_results.json"
        with open(squad_output, 'w', encoding='utf-8') as f:
            json.dump(squad_results, f, indent=2, ensure_ascii=False)
        logger.info(f"SQuAD results saved to {squad_output}")
    
    # Compute and save final metrics
    if all_results:
        logger.info("=" * 50)
        logger.info("Computing Final Metrics")
        logger.info("=" * 50)
        
        final_metrics = compute_final_metrics(all_results)
        
        # Print metrics
        for metric_name, value in final_metrics.items():
            if isinstance(value, float):
                logger.info(f"{metric_name}: {value:.4f}")
            else:
                logger.info(f"{metric_name}: {value}")
        
        # Save all results and metrics
        all_output = output_dir / "all_results.json"
        with open(all_output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        metrics_output = output_dir / "metrics.json"
        with open(metrics_output, 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"All results saved to {all_output}")
        logger.info(f"Final metrics saved to {metrics_output}")
    
    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info("=" * 60)
    logger.info("FASTLORA MODEL EVALUATION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Total evaluation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"All outputs saved in: {output_dir}")
    logger.info(f"Log file: {log_file}")

if __name__ == '__main__':
    main() 