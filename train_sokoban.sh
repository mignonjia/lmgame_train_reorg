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

# using thinking tokens or not
THINK="agent_proxy.enable_think=True trainer.project_name=lmgame_train_with_think"
NO_THINK="agent_proxy.enable_think=False trainer.project_name=lmgame_train_no_think"

# training task
CONFIG_NAME_SOKOBAN="_2_sokoban"
TRAIN_SOKOBAN="trainer.val_only=False trainer.total_training_steps=200 trainer.save_freq=-1 es_manager.train.env_configs.tags=[SimpleSokoban]"

CONFIG_NAME_GSM8K="_7_gsm8k"
TRAIN_GSM8K="trainer.val_only=False trainer.total_training_steps=200 trainer.save_freq=-1 es_manager.train.env_configs.tags=[GSM8K_NoThink_Turn_5]"

CONFIG_NAME_MIX="_8_mix"
TRAIN_MIX="trainer.val_only=False trainer.total_training_steps=200 trainer.save_freq=-1 es_manager.train.env_configs.n_groups=[4,4] es_manager.train.env_configs.tags=[SimpleSokoban,GSM8K_NoThink_Turn_5]"

# evaluation only on sokoban or all tasks
EVAL_SOKOBAN="es_manager.val.env_groups=256 es_manager.val.env_configs.tags=[SimpleSokoban] es_manager.val.env_configs.n_groups=[256]"

EVAL_ALL_TASKS_SMALL_SET="es_manager.val.env_groups=704 es_manager.val.env_configs.tags=[SimpleSokoban,LargerSokoban_Dim_8,Tetris_1,Tetris_2,GSM8K,GSM8K_Turn_5,GSM8K_NoThink,GSM8K_NoThink_Turn_5,Blocksworld3_Text,Blocksworld3_1D,Blocksworld3_Sparse] es_manager.val.env_configs.n_groups=[64,64,64,64,64,64,64,64,64,64,64]"

EVAL_ALL_TASKS_LARGE_SET="es_manager.val.env_groups=2516 es_manager.val.env_configs.tags=[SimpleSokoban,LargerSokoban_Dim_8,Tetris_1,Tetris_2,GSM8K,GSM8K_Turn_5,GSM8K_NoThink,GSM8K_NoThink_Turn_5,Blocksworld3_Text,Blocksworld3_1D,Blocksworld3_Sparse] es_manager.val.env_configs.n_groups=[256,256,256,256,256,256,256,256,156,156,156]"

# Training commands 
python train.py --config-name $CONFIG_NAME_SOKOBAN $SINGLE_GPU $MODEL_HALF_B_INSTRUCT $INIT_MODE $TRAIN_SOKOBAN $THINK $EVAL_SOKOBAN 2>&1 | tee "train_sokoban_half_b.log"
