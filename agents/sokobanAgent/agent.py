# ─────────────────── IMPORTS ───────────────────
from typing import Tuple
from agents import register_agent

# ─────────────────── SOKOBAN AGENT ───────────────────
@register_agent("sokobanAgent")
class SokobanAgent:
    """
    Sokoban Agent for multi-turn RL training.
    Manages game state, history, and LLM interactions.
    """
    
    def __init__(self, config):
        """
        Initialize agent with configuration.
        Sets up system prompt, prompt, max_turns, max_action_per_turn, 
        history, step_count, trajectory, and agent.config
        """
        pass
    
    def initialize_env(self):
        """
        Initialize the Sokoban environment.
        """
        pass
    
    def parse_model_response(self, response: str) -> Tuple[str, str]:
        """
        Parse model response into thought and action.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            tuple: (thought, action_str)
        """
        pass
    
    def get_llm_prompts(self, env_outputs):
        """
        Convert environment outputs to LLM prompts.
        
        Args:
            env_outputs: Environment outputs containing observation, reward, info
            
        Returns:
            DataProto: Formatted prompts for LLM
        """
        pass
    
    def get_env_outputs(self, llm_outputs):
        """
        Process LLM outputs and get environment outputs.
        
        Args:
            llm_outputs: Outputs from LLM
            
        Returns:
            env_outputs: Environment outputs after processing LLM response
        """
        pass
    
    def get_initial_env_outputs(self):
        """
        Get initial environment outputs after first reset.
        
        Returns:
            env_outputs: Initial environment state
        """
        pass
    
    def update_history(self):
        """
        Update agent's interaction history.
        """
        pass
    
    def reset(self):
        """
        Reset agent state for new episode.
        """
        pass
    
    def close(self):
        """
        Clean up agent resources.
        """
        pass
    
    def make_update_row(self):
        """
        Create update row for PPO training.
        
        Returns:
            dict: Training data row
        """
        pass