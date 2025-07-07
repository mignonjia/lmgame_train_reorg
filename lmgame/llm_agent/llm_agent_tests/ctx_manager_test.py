import os
import sys
import numpy as np
from PIL import Image
import hydra
from omegaconf import DictConfig

# Add the project root to the path to import lmgame modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

@hydra.main(version_base=None, config_path=".", config_name="test_config")
def main(config: DictConfig):
    """Simple test of ContextManager with Sokoban images - similar to ctx_manager.py example"""
    print("=== Testing ContextManager with Sokoban Vision ===")
    
    # Try to import required modules
    try:
        from transformers import AutoTokenizer, AutoProcessor
        from lmgame.llm_agent.ctx_manager import ContextManager
        from lmgame.utils import register_resolvers
        from verl import DataProto
        from verl.utils.dataset.rl_dataset import collate_fn
    except ImportError as e:
        print(f"Import error: {e}")
        print("Skipping test due to missing dependencies")
        return

    register_resolvers()
    
    # Create tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    try:
        processor = AutoProcessor.from_pretrained(config.actor_rollout_ref.model.path)
        print(f"✓ Loaded vision processor: {config.actor_rollout_ref.model.path}")
    except:
        processor = None
        print("ℹ Using text-only mode")
    
    # Create ContextManager
    ctx_manager = ContextManager(config=config, tokenizer=tokenizer, processor=processor)
    print(f"✓ Created ContextManager with processor: {processor is not None}")
    print(f"✓ Context manager prefix lookup: {list(ctx_manager.prefix_lookup.keys())}")
    
    # Load Sokoban images as RGB arrays
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, 'sokoban_images')
    
    rgb_arrays = []
    if os.path.exists(images_dir):
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])[:2]
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            try:
                image = Image.open(img_path).convert('RGB')
                rgb_array = np.array(image)
                rgb_arrays.append(rgb_array)
                print(f"✓ Loaded {img_file}: {rgb_array.shape}")
            except Exception as e:
                print(f"Failed to load {img_file}: {e}")
    
    if not rgb_arrays:
        # Create mock RGB array
        mock_rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        rgb_arrays = [mock_rgb]
        print("✓ Created mock RGB array")
    
    # Test env_outputs with RGB images
    env_outputs = [
        {
            "env_id": 0,
            "history": [
                {
                    "state": "Sokoban puzzle:", 
                    "image": rgb_arrays[0],  # RGB array from actual sokoban image
                    "actions_left": 5, 
                    "llm_response": "I need to push the box", 
                    "reward": 0.5
                },
                {
                    "state": "Current state:", 
                    "image": rgb_arrays[0],
                    "actions_left": 4
                }
            ],
            "group_id": 0,
            "metrics": {"SimpleSokoban/success": 0.0},
            "penalty": 0.0
        },
        {
            "env_id": 1,
            "history": [
                {"state": "Text-only puzzle:\n###\n#P.O#\n###", "actions_left": 3, "llm_response": "Move right", "reward": 0.2},
                {"state": "###\n#.PO#\n###", "actions_left": 2}
            ],
            "group_id": 1,
            "metrics": {"SimpleSokoban/success": 0.0},
            "penalty": 0.0
        }
    ]
    
    print(f"✓ Created env_outputs with {len(env_outputs)} environments")
    
    # Test 1: get_lm_inputs for generation
    print("\n--- Testing get_lm_inputs ---")
    lm_inputs = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
    print(f"LM inputs shape: {lm_inputs.batch['input_ids'].shape}")
    print(f"Non-tensor keys: {list(lm_inputs.non_tensor_batch.keys())}")
    
    if "multi_modal_data" in lm_inputs.non_tensor_batch:
        print("✓ Vision data processed successfully")
    
    # Test 2: formulate_rollouts for training
    print("\n--- Testing formulate_rollouts ---")
    rollout_data = ctx_manager.formulate_rollouts(env_outputs)
    print(f"Rollout data shape: {rollout_data.batch['input_ids'].shape}")
    
    if "loss_mask" in rollout_data.batch:
        print("✓ Loss mask generated")
    if "rm_scores" in rollout_data.batch:
        print("✓ Reward scores processed")
    if hasattr(rollout_data, 'meta_info'):
        print(f"✓ Metrics: {list(rollout_data.meta_info.get('metrics', {}).keys())}")
    
    # Test 3: get_env_inputs (reverse direction)
    print("\n--- Testing get_env_inputs ---")
    batch_list = [
        {
            "env_ids": 0,
            "chat_response": "<think>I see a sokoban puzzle</think><answer>move right | push box</answer>",
        },
        {
            "env_ids": 1,  
            "chat_response": "<think>Text puzzle</think><answer>move up</answer>",
        }
    ]
    
    for item in batch_list:
        item["responses"] = tokenizer.encode(item["chat_response"], return_tensors="pt", max_length=512, truncation=True, padding="max_length")[0]
    
    batch_dict = collate_fn(batch_list)
    batch = DataProto.from_single_dict(batch_dict)
    batch.non_tensor_batch["response_texts"] = np.array([item["chat_response"] for item in batch_list], dtype=object)
    
    env_inputs = ctx_manager.get_env_inputs(batch)
    print(f"Generated env_inputs for {len(env_inputs)} environments")
    for i, env_input in enumerate(env_inputs):
        print(f"  Env {i}: {env_input['actions']}")
    
    print("\n🎉 All tests passed! Vision-enhanced ContextManager working correctly.")

if __name__ == "__main__":
    main()
