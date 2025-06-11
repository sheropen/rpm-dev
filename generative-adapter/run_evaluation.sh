#!/bin/bash

# TODO: switch as needed
export CUDA_VISIBLE_DEVICES=1

# Evaluation script for StreamingQA and SQuAD datasets
# Usage: ./run_evaluation.sh

set -e

# Default paths - modify these according to your setup
MODEL_PATH="generative-adaptor/Generative-Adapter-Mistral-7B-Instruct-v0.2"
STREAMINGQA_PATH="../../data/streaminqa_eval.jsonl"
SQUAD_PATH="../../data/squad/dev-v2.0.json"
OUTPUT_DIR="results/$(date +%Y%m%d_%H%M%S)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --streamingqa_path)
            STREAMINGQA_PATH="$2"
            shift 2
            ;;
        --squad_path)
            SQUAD_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max_examples)
            MAX_EXAMPLES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --torch_dtype)
            TORCH_DTYPE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model_path PATH          Path to FastLora model"
            echo "  --streamingqa_path PATH    Path to StreamingQA JSONL file"
            echo "  --squad_path PATH          Path to SQuAD JSON file"
            echo "  --output_dir PATH          Output directory for results"
            echo "  --max_examples N           Maximum number of examples to evaluate"
            echo "  --device DEVICE            Device to use (cuda/cpu)"
            echo "  --torch_dtype DTYPE        Torch dtype (bfloat16/float16/float32)"
            echo "  --help                     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Set default device and torch_dtype if not specified
DEVICE=${DEVICE:-"cuda"}
TORCH_DTYPE=${TORCH_DTYPE:-"bfloat16"}

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "FastLora Model Evaluation"
echo "========================================"
echo "Model Path: $MODEL_PATH"
echo "StreamingQA Path: $STREAMINGQA_PATH"
echo "SQuAD Path: $SQUAD_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Torch Dtype: $TORCH_DTYPE"
echo "========================================"

cd generative-adapter/src

# Build the command
CMD="python eval_streaming_squad.py \
    --model_path '$MODEL_PATH' \
    --output_dir '$OUTPUT_DIR' \
    --device '$DEVICE' \
    --torch_dtype '$TORCH_DTYPE' \
    --squad_path '$SQUAD_PATH' \
    --use_context"

    # TODO:
    # --streamingqa_path '$STREAMINGQA_PATH' \

if [[ -n "$MAX_EXAMPLES" ]]; then
    CMD="$CMD --max_examples $MAX_EXAMPLES"
    echo "✓ Max examples: $MAX_EXAMPLES"
fi

echo ""
echo "Running evaluation..."
echo "Command: $CMD"
echo ""

# Run the evaluation
eval $CMD

echo ""
echo "========================================"
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"

# Display summary if metrics file exists
if [[ -f "$OUTPUT_DIR/metrics.json" ]]; then
    echo ""
    echo "Quick Summary:"
    echo "=============="
    python3 -c "
import json
import sys
try:
    with open('$OUTPUT_DIR/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    print(f'Total Examples: {metrics.get(\"total_examples\", 0)}')
    print(f'Overall F1: {metrics.get(\"average_f1\", 0):.4f}')
    print(f'Overall EM: {metrics.get(\"average_exact_match\", 0):.4f}')
    
    for key, value in metrics.items():
        if '_f1' in key and key != 'average_f1':
            dataset = key.replace('_f1', '')
            em_key = f'{dataset}_exact_match'
            count_key = f'{dataset}_count'
            print(f'{dataset.upper()}: F1={value:.4f}, EM={metrics.get(em_key, 0):.4f}, N={metrics.get(count_key, 0)}')
            
except Exception as e:
    print(f'Error reading metrics: {e}')
"
fi 