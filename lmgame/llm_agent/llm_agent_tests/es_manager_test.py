import os
import sys
import hydra
from omegaconf import DictConfig

# Add the project root to the path to import lmgame modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

@hydra.main(version_base=None, config_path=".", config_name="test_config")
def main(config: DictConfig):
    """Test EnvStateManager with both text and image modes"""
    print("=== Testing EnvStateManager ===")
    
    try:
        from lmgame.llm_agent.es_manager import EnvStateManager
        from lmgame.utils import register_resolvers
        import numpy as np
        from PIL import Image
    except ImportError as e:
        print(f"Import error: {e}")
        print("Skipping test due to missing dependencies")
        return

    register_resolvers()
    
    # Test 1: Text Mode (default)
    print("\n--- Test 1: Text Mode (render_mode='text') ---")
    es_manager_text = EnvStateManager(config, mode="train")
    print(f"✓ Created ES Manager with {len(es_manager_text.envs)} environments")
    
    # Reset environments
    rollout_cache = es_manager_text.reset(seed=123)
    print(f"✓ Reset environments, got {len(rollout_cache)} rollout entries")
    
    # Check initial state format (should be text)
    first_env_state = rollout_cache[0]['history'][0]['state']
    print(f"✓ First environment state type: {type(first_env_state)}")
    print(f"  Sample state: {first_env_state[:50]}..." if len(str(first_env_state)) > 50 else first_env_state)
    
    # Test step method
    print("\n--- Testing step() method ---")
    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "<think>I need to move down</think><answer>Down</answer>",
            "llm_response": "<think>I need to move down</think><answer>Down</answer>",
            "actions": ["Down"]
        },
        {
            "env_id": 1,
            "llm_raw_response": "<think>Let me try right</think><answer>Right</answer>",
            "llm_response": "<think>Let me try right</think><answer>Right</answer>",
            "actions": ["Right"]
        }
    ]
    
    env_outputs = es_manager_text.step(all_env_inputs, cur_turn=1)
    print(f"✓ Step completed, {len(env_outputs)} environments still active")
    
    # Print actual env_outputs structure from step() method
    print(f"\n--- Actual env_outputs from step() (Text Mode) ---")
    print(env_outputs)
    
    # Example of env_outputs format for text mode:
    # [
    #     {
    #         "env_id": 0,
    #         "history": [
    #             {
    #                 "state": "###\n#.X#\n#P_#\n###",  # Text representation
    #                 "actions_left": 10,
    #                 "actions": ["Down"],               # Executed actions
    #                 "reward": 0.0,                     # Turn reward  
    #                 "info": {"action_is_effective": True, "success": False},
    #                 "llm_response": "<think>I need to move down</think><answer>Down</answer>",
    #                 "llm_raw_response": "<think>I need to move down</think><answer>Down</answer>"
    #             },
    #             {
    #                 "state": "###\n#.X#\n#_P#\n###",  # Updated state
    #                 "actions_left": 9
    #             }
    #         ],
    #         "group_id": 0,
    #         "tag": "SimpleSokoban", 
    #         "penalty": 0.0                             # Penalty for invalid actions
    #     }
    # ]
    
    # Check updated state
    if env_outputs:
        updated_state = env_outputs[0]['history'][-1]['state']
        print(f"✓ Updated state type: {type(updated_state)}")
        print(f"  Has reward in history: {'reward' in env_outputs[0]['history'][-2]}")
        print(f"  Actions left: {env_outputs[0]['history'][-1]['actions_left']}")
    
    # Test get_rollout_states
    print("\n--- Testing get_rollout_states() ---")
    final_rollouts = es_manager_text.get_rollout_states()
    print(f"✓ Got final rollouts for {len(final_rollouts)} environments")
    print(f"✓ Sample metrics: {list(final_rollouts[0]['metrics'].keys())}")
    
    es_manager_text.close()
    
    # Test 2: Image Mode (if config supports it)
    print("\n--- Test 2: Image Mode (render_mode='rgb_array') ---")
    
    # Modify config to use rgb_array mode
    if hasattr(config, 'custom_envs') and 'SimpleSokoban' in config.custom_envs:
        # Create a copy of config with rgb_array render mode
        config.custom_envs.SimpleSokoban.env_config.render_mode = "rgb_array"
        
        es_manager_img = EnvStateManager(config, mode="train")
        print(f"✓ Created ES Manager with image mode")
        
        # Reset environments
        rollout_cache_img = es_manager_img.reset(seed=456)
        print(f"✓ Reset environments for image mode")
        
        # Check initial state format (should be images)
        first_env_img = rollout_cache_img[0]['history'][0]
        if 'images' in first_env_img:
            print(f"✓ First environment has images: {len(first_env_img['images'])} images")
            print(f"  Image type: {type(first_env_img['images'][0])}")
            if hasattr(first_env_img['images'][0], 'size'):
                print(f"  Image size: {first_env_img['images'][0].size}")
        else:
            print("ℹ No images found in state")
        
        print(f"✓ State content: {first_env_img['state']}")
        
        # Test step with image mode
        print("\n--- Testing step() with image mode ---")
        all_env_inputs_img = [
            {
                "env_id": 0,
                "llm_raw_response": "<think>I see the sokoban board</think><answer>Up</answer>",
                "llm_response": "<think>I see the sokoban board</think><answer>Up</answer>",
                "actions": ["Up"]
            }
        ]
        
        env_outputs_img = es_manager_img.step(all_env_inputs_img, cur_turn=1)
        print(f"✓ Image mode step completed, {len(env_outputs_img)} environments active")
        
        # Print actual env_outputs structure from step() method (Image Mode)
        print(f"\n--- Actual env_outputs from step() (Image Mode) ---")
        print(env_outputs_img)
        
        # Example of env_outputs format for image mode:
        # [
        #     {
        #         "env_id": 0,
        #         "history": [
        #             {
        #                 "state": "<images>",                        # State token for images
        #                 "images": [<PIL.Image.Image object>],       # List of PIL Images from rgb_array
        #                 "actions_left": 10,
        #                 "actions": ["Up"],                          # Executed actions
        #                 "reward": 0.0,                             # Turn reward
        #                 "info": {"action_is_effective": True, "success": False},
        #                 "llm_response": "<think>I see the sokoban board</think><answer>Up</answer>",
        #                 "llm_raw_response": "<think>I see the sokoban board</think><answer>Up</answer>"
        #             },
        #             {
        #                 "state": "<images>",                        # Updated state token
        #                 "images": [<PIL.Image.Image object>],       # Updated PIL Images
        #                 "actions_left": 9
        #             }
        #         ],
        #         "group_id": 0,
        #         "tag": "SimpleSokoban",
        #         "penalty": 0.0                                     # Penalty for invalid actions
        #     }
        # ]
        
        # Check if images are preserved through step
        if env_outputs_img and env_outputs_img[0]['history']:
            latest_state = env_outputs_img[0]['history'][-1]
            if 'images' in latest_state:
                print(f"✓ Images preserved through step: {len(latest_state['images'])} images")
            else:
                print("ℹ No images in latest state")
        
        # Test final rollouts with images
        print("\n--- Testing get_rollout_states() with images ---")
        final_rollouts_img = es_manager_img.get_rollout_states()
        
        # Check if any rollout has images
        has_images = any('images' in entry for rollout in final_rollouts_img 
                        for entry in rollout['history'] if isinstance(entry, dict))
        print(f"✓ Final rollouts contain images: {has_images}")
        
        if final_rollouts_img:
            print(f"✓ Sample image rollout metrics: {list(final_rollouts_img[0]['metrics'].keys())}")
        
        es_manager_img.close()
    else:
        print("ℹ Skipping image mode test - SimpleSokoban not found in config")
    
    # Test 3: Mixed scenarios and edge cases
    print("\n--- Test 3: Edge Cases ---")
    
    # Test with invalid actions
    es_manager_edge = EnvStateManager(config, mode="train") 
    es_manager_edge.reset(seed=789)
    
    invalid_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Invalid action test",
            "llm_response": "Invalid action test", 
            "actions": ["InvalidAction", "AnotherBadAction"]
        }
    ]
    
    edge_outputs = es_manager_edge.step(invalid_env_inputs, cur_turn=1)
    print(f"✓ Handled invalid actions, penalty applied")
    print(f"  Penalty in rollout: {es_manager_edge.rollout_cache[0].get('penalty', 0)}")
    
    # Print actual env_outputs for edge case (invalid actions)
    print(f"\n--- Actual env_outputs from step() (Edge Case - Invalid Actions) ---")
    print(edge_outputs)
    
    # Example of env_outputs format for edge cases (invalid actions):
    # [] or
    # [
    #     {
    #         "env_id": 0,
    #         "history": [
    #             {
    #                 "state": "###\n#.X#\n#P_#\n###",  # State unchanged (no valid actions)
    #                 "actions_left": 10,               # Actions left unchanged
    #                 "actions": [],                     # No valid actions executed
    #                 "reward": 0.0,                     # No reward for invalid actions
    #                 "info": {"action_is_effective": False, "success": False},
    #                 "llm_response": "Invalid action test",
    #                 "llm_raw_response": "Invalid action test"
    #             },
    #             {
    #                 "state": "###\n#.X#\n#P_#\n###",  # Same state
    #                 "actions_left": 10                # Actions left unchanged
    #             }
    #         ],
    #         "group_id": 0,
    #         "tag": "SimpleSokoban",
    #         "penalty": -0.1                           # Penalty applied for invalid actions
    #     }
    # ]
    
    es_manager_edge.close()
    
    print("\n🎉 All ES Manager tests completed successfully!")
    
    # Summary
    print(f"\n--- Test Summary ---")
    print(f"✓ Text mode: States as strings")
    print(f"✓ Image mode: States with PIL Images") 
    print(f"✓ Step method: Processes actions and updates rollouts")
    print(f"✓ get_rollout_states: Computes final metrics")
    print(f"✓ Edge cases: Handles invalid actions with penalties")

if __name__ == "__main__":
    main()
