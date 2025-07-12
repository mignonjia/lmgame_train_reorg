# ─────────────────── IMPORTS ───────────────────
import random
import yaml
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from agents.agent_utils import Trajectory, parse_model_response
from agents.sokobanAgent.env import SokobanEnv
from agents import register_agent

# ─────────────────── DATA STRUCTURES ───────────────────
@dataclass
class EnvOutput:
    """Simple container for environment outputs that SyncMultiTurnRollout expects."""
    done: bool = False
    state: str = ""
    reward: float = 0.0
    info: Dict[str, Any] = None  # type: ignore
    
    def __post_init__(self):
        if self.info is None:
            self.info = {}

# ─────────────────── SOKOBAN AGENT ───────────────────
@register_agent("sokobanAgent")
class SokobanAgent:
    """
    Sokoban agent that manages environment interactions and conversation history.
    Compatible with SyncMultiTurnRollout interface.
    """
    
    def __init__(self, config, group_id=0, agent_id=0, seed=None, tag=None):
        self.group_id = group_id
        self.agent_id = agent_id
        self.tag = tag
        self.cur_turn = 0
        
        if seed is None:
            self.seed = random.randint(0, 2**32 - 1)
        else:
            self.seed = seed
        
        self.agent_config = config['sokobanAgent']
        self.env_config = config['sokobanEnv']

        self.max_actions_per_turn = self.agent_config['max_actions_per_turn']
        self.max_turns = self.agent_config['max_turns']
        
        self.system_prompt = self.agent_config.get('system_prompt', "You are a helpful AI assistant that solves Sokoban puzzles step by step.")
        self.prompt = self.agent_config.get('prompt', "You are solving the Sokoban puzzle.")
        self.prompt = self._build_enhanced_prompt(self.prompt)
        
        self.turn_prompt_template = """Turn {turn_number}:\nState:\n{state}\n You have {turns_remaining} turns left."""

        self.initialize_env()
        self.reset(self.seed)
        self.trajectory_history = []
        self.messages = []

    def _build_enhanced_prompt(self, base_prompt):
        """Build enhanced prompt with environment info."""
        enhanced_prompt = base_prompt
        
        if self.env_config.get("grid_vocab"):
            symbols = [f"{k}: {v}" for k, v in self.env_config["grid_vocab"].items()]
            grid_vocab = f"\n\nGrid symbols: {', '.join(symbols)}"
            enhanced_prompt += grid_vocab
        
        if self.env_config.get("action_lookup"):
            actions = list(self.env_config["action_lookup"].values())
            action_lookup = f"\n\nAvailable actions: {', '.join(actions)}\nFormat: <answer>Action1 || Action2</answer>"
            enhanced_prompt += action_lookup
        
        return enhanced_prompt

    def initialize_env(self):
        """Initialize the Sokoban environment."""
        self.env = SokobanEnv(self.env_config)

    # ─────────────────── LLM INTERFACE ───────────────────
    def get_llm_prompts(self, env_out):
        """Convert environment outputs to LLM prompts following SyncMultiTurnRollout interface."""
        if env_out.reward != 0.0:
            self.messages.append({"role": "user", "content": f"Reward: {env_out.reward}"})
        
        turn_content = self.turn_prompt_template.format(
            turn_number=self.cur_turn + 1,
            state=env_out.state,
            turns_remaining=self.max_turns - self.cur_turn
        )
        self.messages.append({"role": "user", "content": turn_content})
        
        return self.messages
    
    def get_env_outputs(self, llm_response):
        """Process LLM outputs and get environment outputs."""
        llm_raw_response = llm_response
        
        self.messages.append({"role": "assistant", "content": llm_raw_response})
        self.cur_turn += 1

        processed_llm_response, action_content = parse_model_response(llm_raw_response)
        
        actions = [action.strip() for action in action_content.split('||') if action.strip()]

        obs = self.env.render()
        total_reward = 0
        done = False
        executed_actions = []
        
        action_lookup_reverse = {v: k for k, v in self.env_config['action_lookup'].items()}
        
        for action_str in actions:
            try:
                if action_str in action_lookup_reverse:
                    action = action_lookup_reverse[action_str]
                else:
                    action = int(action_str)
                    
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                executed_actions.append(action)
                if done:
                    break
            except (ValueError, KeyError):
                continue
        
        if self.cur_turn >= self.max_turns:
            done = True
        
        self.update_trajectory_history(
            state=obs,
            actions_left=max(0, self.max_turns - self.cur_turn),
            actions=executed_actions,
            reward=total_reward,
            info={"success": done and total_reward > 0},
            llm_response=processed_llm_response,
            llm_raw_response=llm_raw_response
        )
        
        return EnvOutput(
            done=done,
            state=obs,
            reward=total_reward,
            info={"success": done and total_reward > 0}
        )
    
    def get_initial_env_outputs(self):
        """Get initial environment outputs after first reset."""
        obs = self.env.render()
        return EnvOutput(
            done=False,
            state=obs,
            reward=0.0,
            info={}
        )
    
    # ─────────────────── ROLLOUT STATE COLLECTION ───────────────────
    def get_final_rollout_states(self):
        """Get final rollout states for PPO training."""
        conversation_history = ""
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                conversation_history += f"System: {content}\n"
            elif role == "user":
                conversation_history += f"User: {content}\n"
            elif role == "assistant":
                conversation_history += f"Assistant: {content}\n"
        
        final_response = ""
        for msg in reversed(self.messages):
            if msg["role"] == "assistant":
                final_response = msg["content"]
                break
        
        episode_reward = sum(traj.reward for traj in self.trajectory_history)
        
        row_dict = {
            'conversation_history': conversation_history,
            'final_response': final_response,
            'episode_reward': episode_reward,
            'reward': episode_reward,
            'env_id': self.agent_id,
            'group_id': self.group_id,
            'uid': f"agent_{self.agent_id}_group_{self.group_id}",
            'metrics': {
                'sokobanAgent/success': int(any(traj.info.get('success', False) for traj in self.trajectory_history)),
                'sokobanAgent/turns_taken': len(self.trajectory_history),
                'sokobanAgent/total_reward': episode_reward
            },
            'tag': self.tag or 'sokobanAgent'
        }
        
        return row_dict

    # ─────────────────── TRAJECTORY MANAGEMENT ───────────────────
    def update_trajectory_history(self, state: str, actions_left: int, actions: List[int], 
                                 reward: float, info: Dict[str, Any], llm_response: str, llm_raw_response: str):
        """Update agent's trajectory history."""
        trajectory = Trajectory(
            state=state,
            actions_left=actions_left,
            actions=actions,
            reward=reward,
            info=info,
            llm_response=llm_response,
            llm_raw_response=llm_raw_response
        )
        
        self.trajectory_history.append(trajectory)
        
        return self.trajectory_history
    
    # ─────────────────── LIFECYCLE MANAGEMENT ───────────────────
    def reset(self, seed=None):
        """Reset agent state for new episode."""
        reset_seed = seed if seed is not None else self.seed
        obs = self.env.reset(seed=reset_seed)
        self.cur_turn = 0
        
        self.trajectory_history = []

        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt}
        ]
        
        turn_content = self.turn_prompt_template.format(
            turn_number=self.cur_turn + 1,
            state=obs,
            turns_remaining=self.max_turns - self.cur_turn
        )
        self.messages[-1]["content"] += f"\n{turn_content}"
        
    def close(self):
        """Clean up agent resources."""
        if hasattr(self, 'env') and hasattr(self.env, 'close'):
            self.env.close()