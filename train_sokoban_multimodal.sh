#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Create run_logs directory if it doesn't exist
mkdir -p run_logs

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_PREFIX="run_logs/train_sokoban_multimodal_${TIMESTAMP}"

echo "🚀 Starting multimodal Sokoban training..."
echo "📝 Logs will be saved to: ${LOG_PREFIX}.log"

# Default config is single GPU for vision models (memory intensive)
SINGLE_GPU="system.CUDA_VISIBLE_DEVICES=0 trainer.n_gpus_per_node=1"

# load from scratch or resume from checkpoint
INIT_MODE="trainer.resume_mode=disable"
RESUME_MODE="trainer.resume_mode=auto"

# using thinking tokens or not
THINK="agent_proxy.enable_think=True trainer.project_name=lmgame_multimodal_train_with_think"
NO_THINK="agent_proxy.enable_think=False trainer.project_name=lmgame_multimodal_train_no_think"

# Support for multi-modal training: Vision-specific training task
CONFIG_NAME_SOKOBAN_VISION="_multimodal_sokoban"
TRAIN_SOKOBAN_VISION="trainer.val_only=False trainer.total_training_steps=100 trainer.save_freq=-1 es_manager.train.env_configs.tags=[SimpleSokoban]"

# Support for multi-modal training: Evaluation configurations for Sokoban (similar to original)
EVAL_SOKOBAN_VISION="es_manager.val.env_groups=256 es_manager.val.env_configs.tags=[SimpleSokoban] es_manager.val.env_configs.n_groups=[256]"

# Log training configuration
echo "=== Training Configuration ===" | tee "${LOG_PREFIX}.log"
echo "Timestamp: $(date)" | tee -a "${LOG_PREFIX}.log"
echo "Actor Model: Qwen/Qwen2.5-VL-3B-Instruct (Vision)" | tee -a "${LOG_PREFIX}.log"
echo "Critic Model: Qwen/Qwen2.5-3B-Instruct (Text)" | tee -a "${LOG_PREFIX}.log"
echo "GPU Config: ${SINGLE_GPU}" | tee -a "${LOG_PREFIX}.log"
echo "Init Mode: ${INIT_MODE}" | tee -a "${LOG_PREFIX}.log"
echo "Think Mode: ${THINK}" | tee -a "${LOG_PREFIX}.log"
echo "===============================" | tee -a "${LOG_PREFIX}.log"

# Support for multi-modal training: Multimodal Sokoban training command
echo "🎯 Starting training process..." | tee -a "${LOG_PREFIX}.log"
python train.py --config-name $CONFIG_NAME_SOKOBAN_VISION $SINGLE_GPU $INIT_MODE $TRAIN_SOKOBAN_VISION $THINK $EVAL_SOKOBAN_VISION 2>&1 | tee -a "${LOG_PREFIX}.log"

# Log completion
echo "✅ Training completed at: $(date)" | tee -a "${LOG_PREFIX}.log"
echo "📁 Full log saved to: ${LOG_PREFIX}.log" 