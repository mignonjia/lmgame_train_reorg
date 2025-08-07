#!/bin/bash

# ------ Setup ------
# Get the project root directory (parent of examples/quick_run)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# Change to project root
cd "$PROJECT_ROOT"

# ------ Logging Setup ------
# Create cache directory if it doesn't exist
mkdir -p cache

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="cache/train_qwen_half_b_${TIMESTAMP}.log"

echo "Starting training with quick_train_qwen_halfb_config configuration..."
echo "Logging all outputs to: $LOG_FILE"
echo "Working directory: $(pwd)"

# ------ Environment Setup ------
# Set CUDA device explicitly (verl may ignore this in base.yaml so we explicitly set it here)
export CUDA_VISIBLE_DEVICES=0

# Add lmgamerl to Python path
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Use our config directly - Hydra will automatically merge with base config
python lmgamerl/train.py \
  --config-name "quick_train_qwen_halfb_config" \
  2>&1 | tee "$LOG_FILE"

echo "Training completed. Full log available at: $LOG_FILE" 