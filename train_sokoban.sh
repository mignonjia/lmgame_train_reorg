#!/bin/bash

# ------ Logging Setup ------
# Create cache directory if it doesn't exist
mkdir -p cache

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="cache/train_sokoban_${TIMESTAMP}.log"

echo "Starting Sokoban training..."
echo "Logging all outputs to: $LOG_FILE"

# ------ Run Training with Logging ------
# Redirect both stdout and stderr to log file, while also displaying on console
python train.py --config-name "base" 2>&1 | tee "$LOG_FILE"

echo "Training completed. Full log available at: $LOG_FILE"