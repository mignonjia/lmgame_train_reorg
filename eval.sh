#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Define model
MODEL_7B_INSTRUCT="model_path=Qwen/Qwen2.5-7B-Instruct"
MODEL_HALF_B_INSTRUCT="model_path=Qwen/Qwen2.5-0.5B-Instruct"

# Default config in base.yaml is 8 GPU, use this to run on a single GPU
SINGLE_GPU="system.CUDA_VISIBLE_DEVICES=0 trainer.n_gpus_per_node=1"

# load from scratch or resume from checkpoint
INIT_MODE="trainer.resume_mode=disable"
RESUME_MODE="trainer.resume_mode=auto"

# loading from thinking checkpoint or not
CKPT_THINK="trainer.project_name=train_with_think"
CKPT_NO_THINK="trainer.project_name=train_no_think"

# inference: using thinking tokens or not
EVAL_THINK="agent_proxy.enable_think=True"
EVAL_NO_THINK="agent_proxy.enable_think=False"

# trained task
CONFIG_NAME_SOKOBAN="_2_sokoban"

EVAL_ONLY="trainer.val_only=True trainer.save_freq=-1 trainer.total_training_steps=2"

EVAL_SOKOBAN="es_manager.val.env_groups=256 es_manager.val.env_configs.tags=[SimpleSokoban] es_manager.val.env_configs.n_groups=[256]"

EVAL_ALL_TASKS="es_manager.val.env_groups=2516 actor_rollout_ref.rollout.response_length=1024 es_manager.val.env_configs.tags=[SimpleSokoban,LargerSokoban_Dim_8,Tetris_1,Tetris_2,GSM8K,GSM8K_Turn_5,GSM8K_NoThink,GSM8K_NoThink_Turn_5,Blocksworld3_Text,Blocksworld3_1D,Blocksworld3_Sparse] es_manager.val.env_configs.n_groups=[256,256,256,256,256,256,256,256,156,156,156]"

python train.py --config-name $CONFIG_NAME_SOKOBAN $SINGLE_GPU $MODEL_HALF_B_INSTRUCT $RESUME_MODE $EVAL_ONLY $EVAL_SOKOBAN $CKPT_THINK $EVAL_THINK 2>&1 | tee "eval_half_b.log"

