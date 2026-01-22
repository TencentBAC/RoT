#!/bin/bash

CONFIG="configs/stage2_config_qwen3vl_4b.yaml"
CHECKPOINT="path/to/xxx"
STAGE1_CHECKPOINT="path/to/xxx"
DATA_FILE="data/GSM8k-Aug-NL/example_test_processed.jsonl"
DATASET="gsm8k"
SPLIT="test"
MAX_SAMPLES=20
MAX_NEW_TOKENS=64
TEMPERATURE=0.0
NUM_VISION_TOKENS="32"
STOP_THRESHOLD="0.01"
MODEL_TYPE="v2"
MODE="evaluate"
OUTPUT_FILE="path/to/xxx"
OUTPUT_FORMAT="jsonl"

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --checkpoint PATH          "
    echo ""
    echo "Optional:"
    echo "  --config PATH              "
    echo "  --stage1_checkpoint PATH   "
    echo "  --data_file PATH           "
    echo "  --dataset NAME             "
    echo "  --split train|test         "
    echo "  --max_samples N            "
    echo "  --max_new_tokens N         "
    echo "  --temperature FLOAT        "
    echo "  --num_vision_tokens N      "
    echo "  --stop_threshold FLOAT     "
    echo "  --model_type v1|v2         "
    echo "  --mode evaluate|generate   "
    echo "  --output_file PATH         "
    echo "  --output_format json|jsonl "
    echo ""
    echo "Examples:"
    echo "  # "
    echo "  $0 --checkpoint /path/to/checkpoint --dataset gsm8k --split test"
    echo ""
    echo "  # "
    echo "  $0 --checkpoint /path/to/stage2/checkpoint \\"
    echo "     --stage1_checkpoint /path/to/stage1/checkpoint \\"
    echo "     --config configs/stage2_config.yaml"
    echo ""
    echo "  # "
    echo "  $0 --checkpoint /path/to/checkpoint \\"
    echo "     --mode generate --output_format jsonl"
    echo ""
    echo "  # "
    echo "  $0 --checkpoint /path/to/checkpoint --max_samples 10"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --stage1_checkpoint)
            STAGE1_CHECKPOINT="$2"
            shift 2
            ;;
        --data_file)
            DATA_FILE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --num_vision_tokens)
            NUM_VISION_TOKENS="$2"
            shift 2
            ;;
        --stop_threshold)
            STOP_THRESHOLD="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --output_format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "❌ Error: --checkpoint is required"
    echo ""
    show_usage
    exit 1
fi

if [ -z "$STAGE1_CHECKPOINT" ]; then
    echo "⚠️  Warning: --stage1_checkpoint not provided"
    echo "   Stage 1 checkpoint is required for loading special token embeddings"
    echo "   Evaluation may fail or produce incorrect results without it"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

if [ "$SPLIT" != "train" ] && [ "$SPLIT" != "test" ]; then
    echo "❌ Error: --split must be 'train' or 'test', got: $SPLIT"
    exit 1
fi

if [ "$MODEL_TYPE" != "v1" ] && [ "$MODEL_TYPE" != "v2" ]; then
    echo "❌ Error: --model_type must be 'v1' or 'v2', got: $MODEL_TYPE"
    exit 1
fi

if [ "$MODE" != "evaluate" ] && [ "$MODE" != "generate" ]; then
    echo "❌ Error: --mode must be 'evaluate' or 'generate', got: $MODE"
    exit 1
fi

if [ "$OUTPUT_FORMAT" != "json" ] && [ "$OUTPUT_FORMAT" != "jsonl" ]; then
    echo "❌ Error: --output_format must be 'json' or 'jsonl', got: $OUTPUT_FORMAT"
    exit 1
fi

CMD="CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
    --config \"$CONFIG\" \
    --checkpoint \"$CHECKPOINT\" \
    --stage1_checkpoint \"$STAGE1_CHECKPOINT\" \
    --data_file \"$DATA_FILE\" \
    --dataset \"$DATASET\" \
    --split \"$SPLIT\" \
    --max_samples \"$MAX_SAMPLES\" \
    --max_new_tokens \"$MAX_NEW_TOKENS\" \
    --temperature \"$TEMPERATURE\" \
    --num_vision_tokens \"$NUM_VISION_TOKENS\" \
    --stop_threshold \"$STOP_THRESHOLD\" \
    --model_type \"$MODEL_TYPE\" \
    --mode \"$MODE\" \
    --output_file \"$OUTPUT_FILE\" \
    --output_format \"$OUTPUT_FORMAT\""

if [ -n "$STAGE1_CHECKPOINT" ]; then
    CMD="$CMD --stage1_checkpoint \"$STAGE1_CHECKPOINT\""
fi

if [ -n "$DATA_FILE" ]; then
    CMD="$CMD --data_file \"$DATA_FILE\""
fi

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples \"$MAX_SAMPLES\""
fi

if [ -n "$NUM_VISION_TOKENS" ]; then
    CMD="$CMD --num_vision_tokens \"$NUM_VISION_TOKENS\""
fi

if [ -n "$STOP_THRESHOLD" ]; then
    CMD="$CMD --stop_threshold \"$STOP_THRESHOLD\""
fi

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD --output_file \"$OUTPUT_FILE\""
fi

MODE_DISPLAY=$(echo "$MODE" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')
echo "=========================================="
echo "CoT Compressor - $MODE_DISPLAY Mode"
echo "=========================================="
echo "Checkpoint:       $CHECKPOINT"
echo "Config:           $CONFIG"
if [ -n "$STAGE1_CHECKPOINT" ]; then
    echo "Stage1 Checkpoint: $STAGE1_CHECKPOINT"
fi
if [ -n "$DATA_FILE" ]; then
    echo "Data File:        $DATA_FILE"
else
    echo "Dataset:          $DATASET"
    echo "Split:            $SPLIT"
fi
echo "Model Type:       $MODEL_TYPE"
echo "Mode:             $MODE"
echo "Max New Tokens:   $MAX_NEW_TOKENS"
echo "Temperature:      $TEMPERATURE"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max Samples:      $MAX_SAMPLES"
fi
if [ -n "$NUM_VISION_TOKENS" ]; then
    echo "Vision Tokens:    $NUM_VISION_TOKENS (max limit)"
fi
if [ -n "$STOP_THRESHOLD" ]; then
    echo "Stop Threshold:   $STOP_THRESHOLD"
fi
echo "Output Format:    $OUTPUT_FORMAT"
if [ -n "$OUTPUT_FILE" ]; then
    echo "Output File:      $OUTPUT_FILE"
fi
echo "=========================================="
echo ""
echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
else
    echo ""
    echo "❌ Evaluation failed with exit code $?"
    exit 1
fi
