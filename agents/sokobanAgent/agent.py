# ─────────────────── IMPORTS ───────────────────
from agents import register_agent
from agents.agent_utils import parse_model_response, Trajectory
from .env import SokobanEnv
import re

# ─────────────────── SOKOBAN AGENT ───────────────────
@register_agent("sokobanAgent")
class SokobanAgent:
    """
    Sokoban Agent for multi-turn RL training.
    Manages game state, history, and LLM interactions.
    """
    
    def __init__(self, config, group_id=0, seed=None):
        """
        Initialize agent with configuration.
        Sets up system prompt, prompt, max_turns, max_action_per_turn, 
        history, step_count, trajectory, and agent.config
        """
        self.tag = "sokoban"
        self.group_id = group_id   
        self.seed = seed
        
        self.agent_config = config['sokobanAgent']
        self.env_config = config['sokobanEnv']

        self.max_actions_per_turn = self.agent_config['max_actions_per_turn']
        self.max_turns = self.agent_config['max_turns']
        self.system_prompt = config['system_prompt']
        self.initial_prompt = config['prompt']

        self.env = SokobanEnv(self.env_config) # this is just to initialize the environment, so no need to use initialize_env function
        self.reset(seed)
        self.trajectory = Trajectory()

    
    # def initialize_env(self):
    #     """
    #     Initialize the Sokoban environment.
    #     """
    #     pass
        
    
    def get_llm_prompts(self, obs, reward):
        """
        Convert environment outputs to LLM prompts.
        
        Args:
            env_outputs: Environment outputs containing observation, reward, info
            
        Returns:
            DataProto: Formatted prompts for LLM
        """
        self.cur_turn += 1

        prompt = f"""
        Reward of previous turn: {reward}

        Turn {self.cur_turn}:
        State: {obs}
        You have {self.max_turns - self.cur_turn + 1} turns left, with each turn containing at most {self.max_actions_per_turn} actions.
        """
        
        self.messages.append({"role": "user", "content": prompt})
        
        return self.messages
    
    
    def get_env_outputs(self, llm_response):
        """
        Process LLM outputs and get environment outputs.
        
        Args:
            llm_outputs: Outputs from LLM
            
        Returns:
            env_outputs: Environment outputs after processing LLM response
        """
        
        self.messages.append({"role": "assistant", "content": llm_response})

        thought, actions = parse_model_response(llm_response)

        obs = self.env.render()
        total_reward = 0
        done = False
        executed_actions = []
        for action in actions:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            executed_actions.append(action)
            if done:
                break
        
        return obs, total_reward, done
            
                
    
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
    
    def reset(self, seed=None):
        """
        Reset agent state for new episode.
        """
        obs = self.env.reset(seed=seed)
        self.cur_turn = 1

        prompt = f"""
        {self.initial_prompt}
        Turn {self.cur_turn}:
        {obs}
        You have {self.max_turns - self.cur_turn + 1} turns left, with each turn containing at most {self.max_actions_per_turn} actions.
        """

        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
    
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