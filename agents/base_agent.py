from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

class BaseAgent:
    """
    Abstract base class for agents. Provides high-level method signatures for agent lifecycle, environment interaction, LLM interface, trajectory management, and rollout collection.
    """
    def __init__(self, config, group_id=0, agent_id=0, seed=None, tag=None):
        """Initialize the agent with configuration and identifiers."""
        pass

    def initialize_env(self):
        """Initialize the environment for the agent."""
        pass

    def get_llm_prompts(self, env_out):
        """Convert environment outputs to LLM prompts."""
        pass

    def parse_model_response(self, llm_response, enable_think=True):
        """Parse model response into processed response and action list."""
        pass

    def get_env_outputs(self, llm_response):
        """Process LLM outputs and get environment outputs."""
        pass

    def get_final_rollout_states(self):
        """Get final rollout states for training or evaluation."""
        pass

    def update_trajectory_history(self, state: str, actions_left: int, actions: List[int], 
                                 reward: float, info: Dict[str, Any], llm_response: str, llm_raw_response: str):
        """Update the agent's trajectory history."""
        pass

    def reset(self, seed=None):
        """Reset agent state for a new episode and return initial environment outputs."""
        pass

    def close(self):
        """Clean up agent resources."""
        pass
