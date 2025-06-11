# TODO: switch as needed
export CUDA_VISIBLE_DEVICES=0

cd generative-adapter/src

MODEL_PATH="generative-adaptor/Generative-Adapter-Mistral-7B-Instruct-v0.2"
STREAMINGQA_PATH="../../data/streaminqa_eval.jsonl"
SQUAD_PATH="../../data/squad/dev-v2.0.json"
WMT_DOCS_PATH="../../data/processed_docs/wmt_docs.jsonl"
OUTPUT_DIR="results/$(date +%Y%m%d_%H%M%S)"
DEVICE=${DEVICE:-"cuda"}
TORCH_DTYPE=${TORCH_DTYPE:-"bfloat16"}


python eval_streaming_squad.py \
    --model_path $MODEL_PATH \
    --device $DEVICE \
    --torch_dtype $TORCH_DTYPE \
    --streamingqa_path $STREAMINGQA_PATH \
    --wmt_docs_path $WMT_DOCS_PATH \
    --output_dir $OUTPUT_DIR \
    --mode ga