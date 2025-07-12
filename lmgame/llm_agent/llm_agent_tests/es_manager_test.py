"""
Test for EnvStateManager.get_rollout_states() using Sokoban environment
Author: Assistant
Date: 2025-01-XX
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import sys
import os

# Add the project root to the path to enable imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from lmgame.llm_agent.es_manager import EnvStateManager
from lmgame.env.sokoban.config import SokobanEnvConfig


@dataclass
class MockConfig:
    """Mock configuration for testing EnvStateManager with Sokoban environment"""
    
    @dataclass
    class ESManagerConfig:
        @dataclass
        class TrainConfig:
            env_groups: int = 2  # Number of environment groups
            group_size: int = 4  # Environments per group
            
            @dataclass
            class EnvConfigs:
                tags: Optional[List[str]] = None
                n_groups: Optional[List[int]] = None
                
                def __post_init__(self):
                    if self.tags is None:
                        self.tags = ["SimpleSokoban", "HarderSokoban"]
                    if self.n_groups is None:
                        self.n_groups = [1, 1]  # 1 group of each type
            
            def __post_init__(self):
                self.env_configs = self.EnvConfigs()
        
        def __post_init__(self):
            self.train = self.TrainConfig()
        
        format_penalty: float = 0.1  # Penalty for invalid actions
    
    @dataclass
    class CustomEnvs:
        @dataclass
        class SokobanConfig:
            env_type: str = "sokoban"
            max_actions_per_traj: int = 20
            max_turn: int = 5
            env_config: Optional[Dict[str, Any]] = None
            
            def __post_init__(self):
                if self.env_config is None:
                    self.env_config = {
                        "dim_room": (6, 6),
                        "num_boxes": 1,
                        "max_steps": 50,
                        "search_depth": 5,
                        "render_mode": "text",
                        "action_lookup": {
                            1: "up",
                            2: "down", 
                            3: "left",
                            4: "right"
                        }
                    }
        
        # Both SimpleSokoban and HarderSokoban use the same config for this test
        SimpleSokoban: Optional['MockConfig.CustomEnvs.SokobanConfig'] = None
        HarderSokoban: Optional['MockConfig.CustomEnvs.SokobanConfig'] = None
        
        def __post_init__(self):
            self.SimpleSokoban = self.SokobanConfig()
            # Make HarderSokoban slightly different (more boxes)
            harder_config = {
                "dim_room": (6, 6),
                "num_boxes": 2,  # More boxes for harder version
                "max_steps": 50,
                "search_depth": 5,
                "render_mode": "text",
                "action_lookup": {
                    1: "up",
                    2: "down", 
                    3: "left",
                    4: "right"
                }
            }
            self.HarderSokoban = self.SokobanConfig(
                env_config=harder_config,
                max_actions_per_traj=30
            )
        
        def __getitem__(self, key: str):
            """Make CustomEnvs subscriptable like a dictionary"""
            if key == "SimpleSokoban":
                return self.SimpleSokoban
            elif key == "HarderSokoban":
                return self.HarderSokoban
            else:
                raise KeyError(f"Environment '{key}' not found")
        
        def items(self):
            """Make CustomEnvs support items() method"""
            return [("SimpleSokoban", self.SimpleSokoban), ("HarderSokoban", self.HarderSokoban)]
    
    def __post_init__(self):
        self.es_manager = self.ESManagerConfig()
        self.custom_envs = self.CustomEnvs()


def mock_llm_response_generator(env_id: int, turn: int, available_actions: List[str]) -> Dict[str, Any]:
    """
    Generate mock LLM responses with random actions for testing
    
    Args:
        env_id: Environment ID
        turn: Current turn number
        available_actions: List of available actions for this environment
    
    Returns:
        Dictionary with env_id, llm_response, llm_raw_response, and actions
    """
    # Randomly select 1-3 actions
    num_actions = random.randint(1, 3)
    selected_actions = random.choices(available_actions, k=num_actions)
    
    # Create a mock LLM response
    action_text = ", ".join(selected_actions)
    llm_response = f"I'll move {action_text}"
    llm_raw_response = f"Turn {turn}: Let me analyze the situation and move {action_text}. This should help me progress towards the goal."
    
    return {
        "env_id": env_id,
        "llm_response": llm_response,
        "llm_raw_response": llm_raw_response,
        "actions": selected_actions
    }


def test_get_rollout_states_5_turns():
    """
    Test EnvStateManager.get_rollout_states() over 5 turns with Sokoban environment
    """
    print("=" * 60)
    print("Testing EnvStateManager.get_rollout_states() with Sokoban Environment")
    print("=" * 60)
    
    # Create mock configuration
    config = MockConfig()
    
    # Initialize EnvStateManager
    es_manager = EnvStateManager(config, mode="train")
    
    print(f"Initialized {len(es_manager.envs)} environments:")
    for i, env_entry in enumerate(es_manager.envs):
        print(f"  Env {i}: {env_entry['tag']} (Group {env_entry['group_id']})")
    
    # Reset environments
    print("\n" + "-" * 40)
    print("Resetting environments...")
    rollout_cache = es_manager.reset(seed=42)
    print(f"Reset complete. Initial rollout cache has {len(rollout_cache)} entries")
    
    # Show initial state of first few environments
    print("\nInitial states (first 2 environments):")
    for i in range(min(2, len(rollout_cache))):
        cache = rollout_cache[i]
        print(f"\nEnv {cache['env_id']} ({cache['tag']}):")
        if cache['history']:
            print(cache['history'][-1]['state'])
    
    # Simulate 5 turns of LLM interactions
    max_turns = 5
    available_actions = ["up", "down", "left", "right"]
    
    for turn in range(1, max_turns + 1):
        print(f"\n" + "-" * 40)
        print(f"TURN {turn}")
        print("-" * 40)
        
        # Get current active environments (not done yet)
        active_envs = es_manager.rollout_cache
        if active_envs is None:
            print("No rollout cache available!")
            break
            
        active_env_ids = [cache['env_id'] for cache in active_envs 
                         if not (es_manager.envs[cache['env_id']]['status'].terminated or 
                                es_manager.envs[cache['env_id']]['status'].truncated)]
        
        print(f"Active environments: {active_env_ids}")
        
        if not active_env_ids:
            print("No active environments remaining!")
            break
        
        # Generate mock LLM responses for active environments
        all_env_inputs = []
        for env_id in active_env_ids:
            mock_response = mock_llm_response_generator(env_id, turn, available_actions)
            all_env_inputs.append(mock_response)
            print(f"  Env {env_id}: Actions = {mock_response['actions']}")
        
        # Step the environments
        env_outputs = es_manager.step(all_env_inputs, cur_turn=turn)
        
        print(f"Environments still active after turn {turn}: {len(env_outputs)}")
        
        # Show status of all environments
        for env_entry in es_manager.envs[:4]:  # Show first 4 environments
            status = env_entry['status']
            print(f"  Env {env_entry['env_id']}: terminated={status.terminated}, "
                  f"truncated={status.truncated}, actions={status.num_actions}, "
                  f"rewards={status.rewards}")
    
    # Get final rollout states
    print(f"\n" + "=" * 60)
    print("FINAL ROLLOUT STATES")
    print("=" * 60)
    
    final_rollout_states = es_manager.get_rollout_states()
    
    if final_rollout_states is None:
        print("ERROR: get_rollout_states() returned None!")
        return None
    
    print(f"Total environments: {len(final_rollout_states)}")
    
    # Analyze results for each environment
    for i, rollout_state in enumerate(final_rollout_states):
        print(f"\n" + "-" * 30)
        print(f"Environment {rollout_state['env_id']} ({rollout_state['tag']})")
        print("-" * 30)
        
        # Basic info
        print(f"Group ID: {rollout_state['group_id']}")
        print(f"Penalty: {rollout_state['penalty']}")
        print(f"History length: {len(rollout_state['history'])}")
        
        # Environment status
        env_status = es_manager.envs[i]['status']
        print(f"Final status: terminated={env_status.terminated}, truncated={env_status.truncated}")
        print(f"Total actions taken: {env_status.num_actions}")
        print(f"Turn-wise rewards: {env_status.rewards}")
        print(f"Total reward: {sum(env_status.rewards)}")
        
        # Metrics
        if 'metrics' in rollout_state:
            print(f"Environment metrics:")
            for key, value in rollout_state['metrics'].items():
                print(f"  {key}: {value}")
        
        # Show trajectory (condensed)
        print(f"Trajectory summary:")
        for j, turn_data in enumerate(rollout_state['history']):
            if 'actions' in turn_data:  # This is a turn with actions
                actions_str = ', '.join(str(action) for action in turn_data['actions']) if turn_data['actions'] else 'No actions'
                reward = turn_data.get('reward', 0)
                print(f"  Turn {j}: {actions_str} (reward: {reward})")
        
        # Show final state
        if rollout_state['history']:
            final_state = rollout_state['history'][-1]['state']
            print(f"Final state:")
            print(f"  {final_state[:100]}..." if len(final_state) > 100 else f"  {final_state}")
    print(f"\n" + "=" * 60)
    print(f"FINAL ROLLOUT STATES:\n{final_rollout_states}")
    print(f"\n" + "=" * 60)
    # Summary statistics
    print(f"\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    total_envs = len(final_rollout_states)
    successful_envs = sum(1 for state in final_rollout_states 
                         if state['metrics'].get(f"{state['tag']}/success", 0) > 0)
    terminated_envs = sum(1 for i, state in enumerate(final_rollout_states)
                         if es_manager.envs[i]['status'].terminated)
    
    print(f"Total environments: {total_envs}")
    print(f"Successfully completed: {successful_envs}")
    print(f"Terminated (done): {terminated_envs}")
    print(f"Success rate: {successful_envs/total_envs*100:.1f}%")
    
    # Average metrics by environment type
    env_types = {}
    for state in final_rollout_states:
        tag = state['tag']
        if tag not in env_types:
            env_types[tag] = {'count': 0, 'success': 0, 'total_actions': 0, 'total_reward': 0}
        
        env_types[tag]['count'] += 1
        env_types[tag]['success'] += state['metrics'].get(f"{tag}/success", 0)
        env_types[tag]['total_actions'] += state['metrics'].get(f"{tag}/num_actions", 0)
        
        env_idx = state['env_id']
        env_types[tag]['total_reward'] += sum(es_manager.envs[env_idx]['status'].rewards)
    
    print(f"\nBy environment type:")
    for tag, stats in env_types.items():
        avg_actions = stats['total_actions'] / stats['count']
        avg_reward = stats['total_reward'] / stats['count']
        success_rate = stats['success'] / stats['count'] * 100
        print(f"  {tag}: {stats['count']} envs, {success_rate:.1f}% success, "
              f"avg {avg_actions:.1f} actions, avg {avg_reward:.2f} reward")
    
    # Close environments
    es_manager.close()
    print(f"\nTest completed successfully!")
    
    return final_rollout_states


if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # Run the test
    try:
        final_states = test_get_rollout_states_5_turns()
        if final_states is not None:
            print(f"\n✅ Test passed! Generated {len(final_states)} final rollout states.")
        else:
            print(f"\n❌ Test failed! get_rollout_states() returned None.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
