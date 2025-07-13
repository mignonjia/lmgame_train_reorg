#!/usr/bin/env python3
"""
Script to print the hydra integrated config for train_sokoban
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import os

# Add the current directory to the path so we can import lmgame.utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lmgame.utils import register_resolvers
register_resolvers()

@hydra.main(version_base=None, config_path="config", config_name="_2_sokoban")
def print_config(config: DictConfig) -> None:
    print("=" * 80)
    print("HYDRA INTEGRATED CONFIG FOR TRAIN_SOKOBAN")
    print("=" * 80)
    print()
    
    # Print the config in a readable format
    print(OmegaConf.to_yaml(config))
    
    print("=" * 80)
    print("CONFIG SUMMARY")
    print("=" * 80)
    print(f"Model: {config.model_path}")
    print(f"CUDA Devices: {config.system.CUDA_VISIBLE_DEVICES}")
    print(f"GPUs per node: {config.trainer.n_gpus_per_node}")
    print(f"Project name: {config.trainer.project_name}")
    print(f"Experiment name: {config.trainer.experiment_name}")
    print(f"Total training steps: {config.trainer.total_training_steps}")
    print(f"Enable think: {config.agent_proxy.enable_think}")
    print(f"Train env groups: {config.es_manager.train.env_groups}")
    print(f"Train env tags: {config.es_manager.train.env_configs.tags}")
    print(f"Val env groups: {config.es_manager.val.env_groups}")
    print(f"Val env tags: {config.es_manager.val.env_configs.tags}")
    print(f"PPO mini batch size: {config.ppo_mini_batch_size}")
    print(f"Micro batch size per GPU: {config.micro_batch_size_per_gpu}")
    print("=" * 80)

if __name__ == "__main__":
    print_config() 