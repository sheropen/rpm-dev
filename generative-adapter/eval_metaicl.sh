# TODO: switch as needed
export CUDA_VISIBLE_DEVICES=1

cd generative-adapter/src

MODEL_PATH="generative-adaptor/Generative-Adapter-Mistral-7B-Instruct-v0.2"
OUTPUT_DIR="results/metaicl_eval_$(date +%Y%m%d_%H%M%S)"
DEVICE=${DEVICE:-"cuda"}
TORCH_DTYPE=${TORCH_DTYPE:-"bfloat16"}

TASKS=(
    "quarel"
    "financial_phrasebank" 
    "openbookqa"
    "codah"
    "qasc"
    "glue-mrpc"
    "dream"
    "sick"
    "commonsense_qa"
    "medical_questions_pairs"
    "quartz-with_knowledge"
    "poem_sentiment"
    "quartz-no_knowledge"
    "glue-wnli"
    "climate_fever"
    "ethos-national_origin"
    "ethos-race"
    "ethos-religion"
    "ai2_arc"
    "hate_speech18"
    "glue-rte"
    "supergluecb"
    "superglue-copa"
    "tweet_eval-hate"
    "tweet_eval-stance_atheism"
    "tweet_eval-stance_feminist"
)

# Test with multiple k-shot settings and seeds as per MetaICL evaluation protocol
python eval_metaicl.py \
    --model_path $MODEL_PATH \
    --tasks "${TASKS[@]}" \
    --k_shots 1 2 4 8 16 \
    --seeds 100 13 21 42 87 \
    --device $DEVICE \
    --output_dir $OUTPUT_DIR \
    --methods "fastlora" "icl"

echo "Evaluation complete. Results saved to: $OUTPUT_DIR"