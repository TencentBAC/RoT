#!/bin/bash

echo "=========================================="
echo "Stage 2 Training with DeepSpeed"
echo "Only training lm_head"
echo "=========================================="

cd "$(dirname "$0")"

mkdir -p outputs/checkpoints/stage2
mkdir -p outputs/logs/stage2

NUM_GPUS=${NUM_GPUS:-2}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-configs/deepspeed_config.json}
CONFIG=${CONFIG:-configs/stage2_config_qwen3vl_4b.yaml}
DATASET=${DATASET:-gsm8kaug}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_EPOCHS=${NUM_EPOCHS:-2}
LR=${LR:-2e-5}
SAVE_INTERVAL=${SAVE_INTERVAL:-200}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
MASTER_PORT=${MASTER_PORT:-29502}

while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --deepspeed_config)
            DEEPSPEED_CONFIG="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --save_interval)
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        --stage1_checkpoint)
            STAGE1_CHECKPOINT="$2"
            shift 2
            ;;
        --resume_from_checkpoint)
            RESUME_FROM_CHECKPOINT="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--num_gpus N] [--deepspeed_config PATH] [--config PATH] [--dataset NAME] [--batch_size N] [--num_epochs N] [--lr LR] [--save_interval N] [--stage1_checkpoint PATH] [--resume_from_checkpoint PATH] [--master_port PORT]"
            exit 1
            ;;
    esac
done

if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "Error: DeepSpeed config file not found: $DEEPSPEED_CONFIG"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Number of GPUs: $NUM_GPUS"
echo "  DeepSpeed config: $DEEPSPEED_CONFIG"
echo "  Training config: $CONFIG"
echo "  Dataset: $DATASET"
echo "  Batch size: $BATCH_SIZE"
echo "  Number of epochs: $NUM_EPOCHS"
echo "  Learning rate: $LR"
echo "  Save interval: $SAVE_INTERVAL steps"
echo "  Master port: $MASTER_PORT"
if [ -n "$STAGE1_CHECKPOINT" ]; then
    echo "  Stage 1 checkpoint: $STAGE1_CHECKPOINT"
fi
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    echo "  🔄 Resume from checkpoint: $RESUME_FROM_CHECKPOINT"
fi
echo ""

export MASTER_PORT=$MASTER_PORT
export MASTER_ADDR=${MASTER_ADDR:-localhost}

echo "Starting Stage 2 DeepSpeed training..."
echo "  MASTER_PORT=$MASTER_PORT"
echo "  MASTER_ADDR=$MASTER_ADDR"
DEEPSPEED_ARGS=(
    --num_gpus=$NUM_GPUS
    --master_port=$MASTER_PORT
    scripts/train.py
    --config "$CONFIG"
    --dataset "$DATASET"
    --batch_size "$BATCH_SIZE"
    --num_epochs "$NUM_EPOCHS"
    --lr "$LR"
    --deepspeed_config "$DEEPSPEED_CONFIG"
    --save_interval "$SAVE_INTERVAL"
)

if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    DEEPSPEED_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

conda activate deepseek_3.10
deepspeed "${DEEPSPEED_ARGS[@]}"
