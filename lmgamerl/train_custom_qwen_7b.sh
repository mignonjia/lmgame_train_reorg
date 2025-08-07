#!/bin/bash

# ------ Logging Setup ------
# Create cache directory if it doesn't exist
mkdir -p cache

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="cache/train_custom_qwen_7b_${TIMESTAMP}.log"

echo "Starting training with custom_qwen_7b_b configuration..."
echo "Logging all outputs to: $LOG_FILE"

# ------ Run Training with Logging ------
# Redirect both stdout and stderr to log file, while also displaying on console
# verl may ignore this cuda_visible_devices in base.yaml so we explicitly set it here
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py --config-name "custom_qwen_7b_train" 2>&1 | tee "$LOG_FILE"

echo "Training completed. Full log available at: $LOG_FILE" 