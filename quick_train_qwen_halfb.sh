#!/bin/bash

# Setup project root (script is in project root)
PROJECT_ROOT="$(pwd)"
echo "Project root: $PROJECT_ROOT"

# Setup logging
mkdir -p cache
LOG_FILE="cache/train_qwen_half_b_$(date +"%Y%m%d_%H%M%S").log"

echo "Starting training with quick_train_qwen_halfb_config..."
echo "Logging to: $LOG_FILE"

# Set environment and run training
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

python lmgamerl/train.py --config-name "quick_train_qwen_halfb_config" 2>&1 | tee "$LOG_FILE"

echo "Training completed. Log: $LOG_FILE" 