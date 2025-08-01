# ─────────────────── IMPORTS ───────────────────
import random
import yaml
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from agents.agent_utils import SingleTurnTrajectory, MultiTurnTrajectory, EnvOutput
from agents.base_agent import BaseAgent
from agents.gsm8kAgent.env import GSM8KEnv
from agents import register_agent

# ─────────────────── SOKOBAN AGENT ───────────────────
@register_agent("gsm8kAgent")
class GSM8KAgent(BaseAgent):
    """
    GSM8K agent that manages environment interactions and conversation history.
    Compatible with SyncMultiTurnRollout interface.
    """
    
    def __init__(self, config, group_id=0, agent_id=0, seed=None, tag=None):
        # initialize base agent
        self.group_id = group_id
        self.agent_id = agent_id
        self.tag = tag
        self.cur_turn = 0
        if seed is None:
            self.seed = random.randint(0, 2**32 - 1)
        else:
            self.seed = seed
        self.agent_config = config['agent_config']
        self.env_config = config['env_config']

        # handle config hyperparameters
        self.max_turns = self.agent_config.get('max_turns', 1)
        self.max_actions_all_turns = self.agent_config.get('max_actions_all_turns', 1)
        self.max_actions_per_turn = self.agent_config.get('max_actions_per_turn', 1)
        self.max_tokens = self.agent_config.get('max_tokens', 100)
        self.format_penalty = self.agent_config.get('format_penalty', -0.1)
        self.enable_think = self.agent_config.get('enable_think', True)
        self.system_prompt = self.agent_config.get('system_prompt', "You are a helpful AI assistant that solves Sokoban puzzles step by step.")
        self.prompt = self.agent_config.get('prompt', "You are solving the Sokoban puzzle.")

        if self.enable_think:
            self.turn_prompt_template = """Turn {turn_number}:\nState:\n{state}\nYou have {actions_remaining} actions remaining. Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: {max_tokens} tokens.\n"""
        else:
            self.turn_prompt_template = """Turn {turn_number}:\nState:\n{state}\nYou have {actions_remaining} actions remaining. Always output: <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: {max_tokens} tokens.\n"""

        self.initialize_env()
        self.trajectory_history = MultiTurnTrajectory(max_length=self.max_turns)
        self.raw_response_list = []  # Store all raw LLM responses for debugging
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt}
        ]
        self.total_actions_consumed = 0
        self.penalty = 0.0  # Track accumulated penalty
    

    def initialize_env(self):
        """Initialize the GSM8K environment."""
        self.env = GSM8KEnv(self.env_config)

