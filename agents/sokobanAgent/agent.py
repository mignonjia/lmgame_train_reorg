#TODO: Align with reported performance in the sokoban rl training
#TODO: be careful with the parse_response logic, which may be associated with format penality.
# ─────────────────── IMPORTS ───────────────────
import random
import yaml
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from agents.agent_utils import Trajectory
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
        
        # Handle both old and new config formats
        self.agent_config = config['agent_config']
        self.env_config = config['env_config']

        self.max_turns = self.agent_config.get('max_turns', 5)
        self.max_actions_all_turns = self.agent_config.get('max_actions_all_turns', 10)
        self.max_actions_per_turn = self.agent_config.get('max_actions_per_turn', 5)
        self.format_penalty = self.agent_config.get('format_penalty', 0.0)
        self.enable_think = self.agent_config.get('enable_think', True)
        
        self.system_prompt = self.agent_config.get('system_prompt', "You are a helpful AI assistant that solves Sokoban puzzles step by step.")
        self.prompt = self.agent_config.get('prompt', "You are solving the Sokoban puzzle.")
        self.prompt = self._build_enhanced_prompt(self.prompt)
        
        if self.enable_think:
            self.turn_prompt_template = """Turn {turn_number}:\nState:\n{state}\nYou have {turns_remaining} turns left. ALWAYS respond in format: <think>reasoning</think><answer>actions</answer>
                                        Example: <think>I need to move right to reach the box, then push it up to the target.</think><answer>Right || Up</answer>"""
        else:
            self.turn_prompt_template = """Turn {turn_number}:\nState:\n{state}\nYou have {turns_remaining} turns left. ALWAYS respond in format: <answer>actions</answer>
                                        Example: <answer>Right || Up</answer>"""

        self.initialize_env()
        self.trajectory_history = []
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt}
        ]
        self.total_actions_consumed = 0
        self.penalty = 0.0  # Track accumulated penalty

    def _build_enhanced_prompt(self, base_prompt):
        """Build enhanced prompt with environment info and emphatic format instructions."""
        enhanced_prompt = base_prompt
        
        if self.env_config.get("grid_vocab"):
            symbols = [f"{k}: {v}" for k, v in self.env_config["grid_vocab"].items()]
            grid_vocab = f"\nThe meaning of each symbol in the state is:\n {', '.join(symbols)}"
            enhanced_prompt += grid_vocab
        
        if self.env_config.get("action_lookup"):
            actions = list(self.env_config["action_lookup"].values())
            action_lookup_str = "\nYour available actions are:\n" + ", ".join(actions)
            enhanced_prompt += action_lookup_str
            
        return enhanced_prompt

    def initialize_env(self):
        """Initialize the Sokoban environment."""
        self.env = SokobanEnv(self.env_config)

    def _debug_print_messages(self, context=""):
        """Helper method to systematically print messages for debugging."""
        print(f"\n{'='*80}")
        print(f"DEBUG MESSAGES [{context}] - Agent {self.agent_id}, Turn {self.cur_turn}")
        print(f"{'='*80}")
        
        if not hasattr(self, 'messages') or not self.messages:
            print("❌ ERROR: No messages found!")
            return
        
        print(f"Total messages: {len(self.messages)}")
        print(f"Enable think: {self.enable_think}")
        print(f"Current turn: {self.cur_turn}")
        print(f"Max turns: {self.max_turns}")
        print(f"Actions consumed: {self.total_actions_consumed}/{self.max_actions_all_turns}")
        print("-" * 80)
        
        for i, msg in enumerate(self.messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            print(f"Message {i+1}: [{role.upper()}]")
            
            # Truncate very long content for readability
            content_preview = content
                
            print(f"  Content: {repr(content_preview)}")
            print(f"  Length: {len(content)} chars")
            
            # Special handling for assistant messages to show parsing
            if role == 'assistant':
                try:
                    processed_response, actions = self.parse_model_response(content, self.enable_think)
                    print(f"  Parsed actions: {actions}")
                    print(f"  Action count: {len(actions)}")
                except Exception as e:
                    print(f"  ❌ Parse error: {e}")
            
            print("-" * 40)
        
        # Print current Sokoban layout
        current_layout = self.env.render()
        print(f"Current Sokoban Layout:\n{current_layout}")
        print(f"{'='*80}\n")

    # ─────────────────── LLM INTERFACE ───────────────────
    def get_llm_prompts(self, env_out):
        """Convert environment outputs to LLM prompts following SyncMultiTurnRollout interface."""
        
        # ✅ DEFENSIVE CHECK: Ensure messages are initialized
        if not hasattr(self, 'messages') or not self.messages:
            self.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.prompt}
            ]
        
        if env_out.reward != 0.0:
            reward_msg = {"role": "user", "content": f"Reward: {env_out.reward}"}
            self.messages.append(reward_msg)
        
        turn_content = self.turn_prompt_template.format(
            turn_number=self.cur_turn + 1,
            state=env_out.state,
            turns_remaining=self.max_turns - self.cur_turn
        )
        turn_msg = {"role": "user", "content": turn_content}
        self.messages.append(turn_msg)
        
        # Validate final messages before returning
        if not self.messages:
            # Emergency fallback
            self.messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Please respond appropriately."}
            ]

        # ✅ DEBUG: Print messages systematically
        self._debug_print_messages(f"GET_LLM_PROMPTS")
        
        return self.messages

    def parse_model_response(self, llm_response, enable_think=True):
        """
        Parse model response into processed llm_response and action list.
        Simple parsing that handles enable_think cases and limits actions to max_actions_per_turn.
        
        Args:
            llm_response: Raw LLM response string
            enable_think: Whether to expect <think> tags
            
        Returns:
            Tuple[str, List[str]]: (processed_llm_response, actions_list)
        """
        import re
        
        # Define pattern based on enable_think
        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if enable_think else r'<answer>(.*?)</answer>'
        match = re.search(pattern, llm_response, re.DOTALL)
        
        if not match:
            # No valid pattern found, return original response with empty actions
            processed_response, actions = llm_response, []
        else:
            if enable_think:
                think_content, action_content = match.group(1), match.group(2)
            else:
                think_content, action_content = "", match.group(1)
            
            # Clean up special tokens
            special_tokens = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]
            for special_token in special_tokens:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
            
            # Parse actions using || separator
            actions = [action.strip() for action in action_content.split("||") if action.strip()]
            
            # Limit actions to max_actions_per_turn
            if len(actions) > self.max_actions_per_turn:
                actions = actions[:self.max_actions_per_turn]  # Only the first MAX_ACTIONS actions are kept
                action_content = " || ".join(actions)
            
            # Reconstruct properly formatted response
            if enable_think:
                processed_response = f"<think>{think_content}</think><answer>{action_content}</answer>"
            else:
                processed_response = f"<answer>{action_content}</answer>"
        
        return processed_response, actions

    def get_env_outputs(self, llm_response):
        """Process LLM outputs and get environment outputs."""
        llm_raw_response = llm_response
        
       
        self.cur_turn += 1

        processed_llm_response, actions = self.parse_model_response(llm_raw_response, enable_think=self.enable_think)

        self.messages.append({"role": "assistant", "content": processed_llm_response})

        obs = self.env.render()
        total_reward = 0
        done = False
        executed_actions = []
        info = {}  # Initialize info dictionary
        
        action_lookup_reverse = {v: k for k, v in self.env_config['action_lookup'].items()}
        
        valid_actions = []
        invalid_actions = []
        
        for action_str in actions:
            try:
                if action_str in action_lookup_reverse:
                    action = action_lookup_reverse[action_str]
                    # ✅ FAULT TOLERANCE: Validate action is in expected range
                    if action in self.env_config['action_lookup']:
                        valid_actions.append(action)
                    else:
                        invalid_actions.append(action_str)
                else:
                    action = int(action_str)
                    # ✅ FAULT TOLERANCE: Validate numeric action is in expected range  
                    if action in self.env_config['action_lookup']:
                        valid_actions.append(action)
                    else:
                        invalid_actions.append(action_str)
            except (ValueError, KeyError, TypeError) as e:
                invalid_actions.append(action_str)
                continue
        
        # Apply penalty for invalid actions
        if invalid_actions or len(valid_actions) != len(actions):
            self.penalty += self.format_penalty
        
        # Execute valid actions with fault tolerance
        for action in valid_actions:
            try:
                obs, reward, done, step_info = self.env.step(action)
                total_reward += reward
                executed_actions.append(action)
                info.update(step_info)  # Update info with step info
                if done:
                    break
            except Exception as e:
                # ✅ FAULT TOLERANCE: Handle any environment step errors
                print(f"Warning: Agent {self.agent_id} step failed for action {action}: {e}")
                # Continue with next action instead of crashing
                continue
        
        # Update total actions consumed
        self.total_actions_consumed += len(executed_actions)
        
        # Calculate actions left based on max_actions_all_turns
        actions_left = max(0, self.max_actions_all_turns - self.total_actions_consumed)
        
        # Check if done due to max turns or max actions
        if self.cur_turn >= self.max_turns or self.total_actions_consumed >= self.max_actions_all_turns:
            done = True
        
        self.update_trajectory_history(
            state=obs,
            actions_left=actions_left,
            actions=executed_actions,
            reward=total_reward,
            info=info,
            llm_response=processed_llm_response,
            llm_raw_response=llm_raw_response
        )
        
        return EnvOutput(
            done=done,
            state=obs,
            reward=total_reward,
            info=info
        )
    
    # ─────────────────── ROLLOUT STATE COLLECTION ───────────────────
    def get_final_rollout_states(self):
        """Get final rollout states for PPO training."""
        history = []
        for traj in self.trajectory_history:
            history_entry = {
                'state': traj.state,
                'actions_left': traj.actions_left,
                'actions': traj.actions,
                'reward': traj.reward,
                'info': traj.info,
                'llm_response': traj.llm_response,
                'llm_raw_response': traj.llm_raw_response
            }
            history.append(history_entry)
        
        metrics = {}
        
        success_values = [traj.info.get('success', False) for traj in self.trajectory_history]
        metrics[f'{self.tag or "sokobanAgent"}/success'] = float(any(success_values))
        
        total_actions = sum(len(traj.actions) for traj in self.trajectory_history)
        metrics[f'{self.tag or "sokobanAgent"}/num_actions'] = total_actions
        
        action_is_effective_values = [traj.info.get('action_is_effective', False) for traj in self.trajectory_history]
        if action_is_effective_values:
            metrics[f'{self.tag or "sokobanAgent"}/action_is_effective'] = sum(action_is_effective_values) / len(action_is_effective_values)
        else:
            metrics[f'{self.tag or "sokobanAgent"}/action_is_effective'] = 0.0
        
        action_is_valid_values = [traj.info.get('action_is_valid', True) for traj in self.trajectory_history]
        if action_is_valid_values:
            metrics[f'{self.tag or "sokobanAgent"}/action_is_valid'] = sum(action_is_valid_values) / len(action_is_valid_values)
        else:
            metrics[f'{self.tag or "sokobanAgent"}/action_is_valid'] = 1.0
        
        if self.trajectory_history:
            last_traj = self.trajectory_history[-1]
            if 'metrics' in last_traj.info:
                for key, value in last_traj.info['metrics'].items():
                    metrics[key] = value
        
        row_dict = {
            'env_id': self.agent_id,
            'history': history,
            'group_id': self.group_id,
            'tag': self.tag or 'sokobanAgent',
            'metrics': metrics,
            'penalty': self.penalty
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
        """Reset agent state for new episode and return initial environment outputs."""
        # ✅ FIX: Implement group-based seeding following reference implementation
        # Agents within the same group should have the same environment (same seed)
        # Different groups should have different environments (different seeds)
        if seed is None:
            # Generate a unique seed only if no seed provided
            reset_seed = random.randint(0, 1000000)
        else:
            # Use the provided group seed directly - all agents in same group get same seed
            reset_seed = seed
            
        obs = self.env.reset(seed=reset_seed)
        
        self.cur_turn = 0
        
        self.trajectory_history = []
        self.total_actions_consumed = 0
        self.penalty = 0.0  # Reset penalty for new episode

        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt}
        ]
        
        # Return initial environment outputs for the rollout loop
        return EnvOutput(
            done=False,
            state=obs,
            reward=0.0,
            info={}
        )

    def close(self):
        """Clean up agent resources."""
        if hasattr(self, 'env') and hasattr(self.env, 'close'):
            self.env.close()