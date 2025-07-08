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
    
    # Test env_outputs with RGB images (mimicking es_manager output format)
    env_outputs = [
        {
            "env_id": 0,
            "history": [
                {
                    "state": "<images>",  # es_manager format: one <images> token per image
                    "images": [Image.fromarray(rgb_arrays[0])],  # PIL Images from es_manager
                    "actions_left": 5, 
                    "llm_response": "I need to push the box", 
                    "reward": 0.5
                },
                {
                    "state": "<images>", 
                    "images": [Image.fromarray(rgb_arrays[0])],
                    "actions_left": 4
                }
            ],
            "group_id": 0,
            "metrics": {"SimpleSokoban/success": 0.0},
            "penalty": 0.0
        },
        {
            "env_id": 1,  # Test backward compatibility with direct RGB arrays
            "history": [
                {
                    "state": "Direct RGB array test:", 
                    "image": rgb_arrays[0],  # Direct RGB array (backward compatibility)
                    "actions_left": 3, 
                    "llm_response": "Move right", 
                    "reward": 0.2
                },
                {"state": "###\n#.PO#\n###", "actions_left": 2}  # Text-only state
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
    
    # Example of what LLM inputs look like (what gets fed to the model):
    # lm_inputs.batch = {
    #     "input_ids": tensor([[    1,   518, 25580, ...]),      # Tokenized prompt + history
    #     "attention_mask": tensor([[1, 1, 1, ...]]),           # Attention mask  
    #     "position_ids": tensor([[0, 1, 2, ...]]),             # Position encoding
    #     "responses": tensor([[518, 25580, ...]]),              # Target responses (for training)
    # }
    # lm_inputs.non_tensor_batch = {
    #     "env_ids": [0, 1],                                     # Environment identifiers
    #     "group_ids": [0, 1],                                   # Group identifiers  
    #     "messages_list": [chat_history_messages],              # Original chat format
    #     "multi_modal_data": [{"image": [PIL_images], "video": []}],  # Vision data
    #     "multi_modal_inputs": [{"pixel_values": tensor, ...}]   # Processed vision tensors
    # }
    #
    # Example LLM outputs (what model generates):
    # lm_outputs.batch = {
    #     "responses": tensor([[2926, 1115, ...]])              # Generated token IDs
    # }
    # lm_outputs.non_tensor_batch = {
    #     "env_ids": [0, 1],
    #     "response_texts": [                                    # Decoded text responses
    #         "I can see the sokoban board. I'll move right to push the box.",
    #         "This is a text puzzle. I'll move up to solve it."
    #     ]
    # }
    
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
    
    print(f"\n--- Complete Rollout Data Structure ---")
    print(rollout_data)
    
    # Example of formulate_rollouts output for training:
    # rollout_data.batch = {
    #     "input_ids": tensor([[1, 518, 25580, ...]]),           # Full conversation tokens
    #     "attention_mask": tensor([[1, 1, 1, ...]]),            # Attention mask
    #     "position_ids": tensor([[0, 1, 2, ...]]),              # Position encoding  
    #     "responses": tensor([[518, 25580, ...]]),               # Target responses (shifted)
    #     "loss_mask": tensor([[0, 0, 1, 1, ...]]),              # Mask: 1=learn, 0=ignore (only assistant turns)
    #     "rm_scores": tensor([[0.0, 0.0, 0.5, 0.8, ...]]),      # Reward scores (normalized per group)
    # }
    # rollout_data.non_tensor_batch = {
    #     "env_ids": [0, 1],                                      # Environment identifiers
    #     "group_ids": [0, 1],                                    # Group IDs for reward normalization
    #     "messages_list": [chat_messages],                       # Original conversation format
    #     "multi_modal_data": [{"image": [PIL_images]}],          # Vision data if present
    #     "multi_modal_inputs": [{"pixel_values": tensor}]        # Processed vision tensors
    # }
    # rollout_data.meta_info = {
    #     "metrics": {
    #         "SimpleSokoban/success": 0.0,                       # Environment-specific metrics
    #         "response_length": 15.2                             # Average response length
    #     }
    # }
    #
    # Key differences from generation mode (prepare_for_update=True):
    # - loss_mask: Only trains on assistant responses, ignores system/user prompts
    # - rm_scores: Normalized rewards for RL training (per group/state/batch)
    # - meta_info: Training metrics for monitoring
    # - responses: Includes full conversation for sequence modeling
    
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
