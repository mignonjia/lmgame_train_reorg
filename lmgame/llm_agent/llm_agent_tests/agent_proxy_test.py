#!/usr/bin/env python3
"""
Test for LLMAgentProxy.rollout() method from agent_proxy.py.
Tests the actual rollout method using EnhancedVllmWrapperWg for both text and image modes.
"""

import os
import sys
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Add the project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from lmgame.llm_agent.ctx_manager import ContextManager
from lmgame.llm_agent.es_manager import EnvStateManager
from lmgame.llm_agent.agent_proxy import LLMAgentProxy
from transformers import AutoTokenizer, AutoProcessor
from verl import DataProto

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("❌ VLLM not available - tests cannot run")
    sys.exit(1)

# Enhanced VLLM Wrapper that handles both text and vision inputs
# todo: llm.generate() doesn't support vision inputs, so we need to use another method.
class EnhancedVllmWrapperWg:
    """Enhanced VLLM wrapper that supports both text and multimodal (vision) inputs"""
    
    def __init__(self, config, tokenizer, processor=None):
        if not VLLM_AVAILABLE:
            raise ImportError("VLLM is not available")
            
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        
        model_name = config.actor_rollout_ref.model.path
        ro_config = config.actor_rollout_ref.rollout
        
        # Check if this is a vision model
        is_vision_model = "VL" in model_name
        
        print(f"🔍 Vision model detected: {is_vision_model}")
        
        # Initialize VLLM with enhanced parameters for multimodal support
        vllm_kwargs = {
            'model': model_name,
            'tensor_parallel_size': getattr(ro_config, 'tensor_model_parallel_size', 1),
            'dtype': getattr(ro_config, 'dtype', 'auto'),
            'enforce_eager': getattr(ro_config, 'enforce_eager', True),
            'gpu_memory_utilization': getattr(ro_config, 'gpu_memory_utilization', 0.9),
            'disable_custom_all_reduce': True,
            'skip_tokenizer_init': getattr(ro_config, 'skip_tokenizer_init', False),
            'max_model_len': getattr(ro_config, 'max_model_len', 4096),
            'disable_log_stats': getattr(ro_config, 'disable_log_stats', True),
            'max_num_batched_tokens': getattr(ro_config, 'max_num_batched_tokens', 1024),
            'enable_chunked_prefill': getattr(ro_config, 'enable_chunked_prefill', False),
            'enable_prefix_caching': getattr(ro_config, 'enable_prefix_caching', False),
        }
        
        # Add vision-specific parameters only for vision models
        if is_vision_model:
            vllm_kwargs.update({
                'trust_remote_code': True,  # Required for vision models
                'disable_mm_preprocessor_cache': getattr(ro_config, 'disable_mm_preprocessor_cache', True),
            })
            print(f"🔍 Vision model detected, adding vision-specific parameters")
        else:
            print(f"📝 Text-only model detected, using standard parameters")
        
        print(f"🚀 Initializing VLLM with model: {model_name}")
        print(f"💾 GPU memory utilization: {vllm_kwargs['gpu_memory_utilization']}")
        print(f"📏 Max model length: {vllm_kwargs['max_model_len']}")
        
        try:
            self.llm = LLM(**vllm_kwargs)
            print("✅ VLLM LLM initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize VLLM: {e}")
            raise
        
        # Set up sampling parameters
        val_kwargs = getattr(ro_config, 'val_kwargs', {})
        self.sampling_params = SamplingParams(
            max_tokens=getattr(ro_config, 'response_length', 128),
            temperature=getattr(val_kwargs, 'temperature', 0.0),
            top_p=getattr(val_kwargs, 'top_p', 1.0),
            top_k=getattr(val_kwargs, 'top_k', -1),
        )
        print(f"📋 Sampling params: max_tokens={self.sampling_params.max_tokens}, temperature={self.sampling_params.temperature}")

    def generate_sequences(self, lm_inputs: DataProto):
        """
        Generate sequences supporting both text-only and multimodal inputs.
        Uses the multimodal data already prepared by ctx_manager.
        """
        input_ids = lm_inputs.batch['input_ids']
        
        # Check if multimodal data is present and has actual images
        has_multimodal = (
            'multi_modal_data' in lm_inputs.non_tensor_batch and
            'multi_modal_inputs' in lm_inputs.non_tensor_batch
        )
        
        # Check if any environment actually has images
        has_actual_images = False
        if has_multimodal:
            multi_modal_data_list = lm_inputs.non_tensor_batch['multi_modal_data']
            has_actual_images = any(
                len(mmd.get("image", [])) > 0 for mmd in multi_modal_data_list
            )
        
        if has_actual_images and self.processor is not None:
            print("🖼️  Processing TRUE multimodal inputs with REAL images")
            print(f"🔍 Found images in {sum(1 for mmd in multi_modal_data_list if len(mmd.get('image', [])) > 0)} environments")
            
            # Prepare VLLM inputs with proper multimodal format
            vllm_inputs = []
            multi_modal_data_list = lm_inputs.non_tensor_batch['multi_modal_data']
            
            # Try to fetch raw_prompt_ids (with vision placeholders) if available
            raw_prompt_ids_list = None
            if 'multi_modal_inputs' in lm_inputs.non_tensor_batch:
                mm_inputs_arr = lm_inputs.non_tensor_batch['multi_modal_inputs']
                raw_prompt_ids_list = [mm_dict.get('raw_prompt_ids') if isinstance(mm_dict, dict) else None for mm_dict in mm_inputs_arr]

            for i in range(input_ids.shape[0]):
                # Prefer raw_prompt_ids (preserves <|vision_start|> etc.), otherwise fall back to tokenizer ids
                if raw_prompt_ids_list and raw_prompt_ids_list[i] is not None:
                    prompt_token_ids = raw_prompt_ids_list[i]
                else:
                    prompt_token_ids = input_ids[i].tolist()

                env_mm_data = multi_modal_data_list[i]

                if len(env_mm_data.get("image", [])) > 0:
                    vllm_input = {
                        "prompt_token_ids": prompt_token_ids,
                        "multi_modal_data": {
                            "image": env_mm_data["image"]
                        }
                    }
                    print(f"🖼️  Environment {i}: Creating multimodal input with {len(env_mm_data['image'])} images (using raw_prompt_ids={raw_prompt_ids_list is not None})")
                else:
                    vllm_input = {"prompt_token_ids": prompt_token_ids}
                    print(f"📝 Environment {i}: Creating text-only input")

                vllm_inputs.append(vllm_input)
            
            try:
                # Generate with proper multimodal inputs using token IDs
                outputs = self.llm.generate(vllm_inputs, sampling_params=self.sampling_params)
                print("✅ TRUE multimodal generation successful - images were processed with preserved <image> tokens!")
            except Exception as e:
                print(f"❌ Multimodal generation failed: {e}")
                # print("🔄 Falling back to text-only generation...")
                # # Fallback to text-only
                # input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
                # input_texts = [text.replace("<|endoftext|>", "") for text in input_texts]
                # outputs = self.llm.generate(input_texts, sampling_params=self.sampling_params)
                raise e  # Re-raise the exception to see the exact error
        else:
            print("📝 Processing text-only inputs (no images found)")
            # Standard text-only generation
            input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            input_texts = [text.replace("<|endoftext|>", "") for text in input_texts]
            outputs = self.llm.generate(input_texts, sampling_params=self.sampling_params)
        
        # Extract generated texts
        texts = [output.outputs[0].text for output in outputs]
        print(f"📝 Generated {len(texts)} responses")
        
        # Create output DataProto
        lm_outputs = DataProto()
        lm_outputs.non_tensor_batch = {
            'response_texts': texts,
            'env_ids': lm_inputs.non_tensor_batch['env_ids'],
            'group_ids': lm_inputs.non_tensor_batch['group_ids']
        }
        lm_outputs.meta_info = lm_inputs.meta_info
        
        return lm_outputs

def test_agent_proxy_with_text():
    """Test agent proxy rollout with text mode (render_mode='text')"""
    print("\n" + "="*60)
    print("🧪 AGENT PROXY TEXT MODE TEST")
    print("="*60)
    
    # Set up environment for VLLM (similar to agent_proxy.py)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    # Load test configuration
    config_dir = os.path.dirname(os.path.abspath(__file__))
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="test_config.yaml")
    
    # Use smaller text-only model for text mode tests
    config.actor_rollout_ref.model.path = "Qwen/Qwen2.5-0.5B"
    
    # Ensure text mode
    config.custom_envs.SimpleSokoban.env_config.render_mode = "text"
    
    print(f"📝 Configuration loaded from: {config_dir}")
    print(f"🤖 Using text-only model: {config.actor_rollout_ref.model.path}")
    print(f"🎮 Environment mode: {config.custom_envs.SimpleSokoban.env_config.render_mode}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    print(f"🤖 Tokenizer loaded: {config.actor_rollout_ref.model.path}")
    
    # Create EnhancedVllmWrapperWg for real model generation
    print("🚀 Creating EnhancedVllmWrapperWg for text generation...")
    actor_wg = EnhancedVllmWrapperWg(config, tokenizer)
    print("✅ EnhancedVllmWrapperWg created successfully!")
    
    agent_proxy = LLMAgentProxy(config, actor_wg, tokenizer, processor=None)  # Text mode - no processor needed
    print("🎯 LLMAgentProxy created successfully")
    
    # Create rollout DataProto (similar to agent_proxy.py main)
    rollout_dataproto = DataProto(
        batch=None, 
        non_tensor_batch=None, 
        meta_info={
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
            'recompute_log_prob': False,
            'do_sample': False,
            'validate': True
        }
    )
    
    print("📦 DataProto created")
    
    # Test the actual rollout method from LLMAgentProxy
    print(f"\n🚀 Testing agent_proxy.rollout() method...")
    rollouts = agent_proxy.rollout(rollout_dataproto, val=True)
    
    print(f"\n📊 ROLLOUT RESULTS:")
    print(f"✓ Batch input_ids shape: {rollouts.batch['input_ids'].shape}")
    print(f"✓ Batch loss_mask shape: {rollouts.batch['loss_mask'].shape}")
    print(f"✓ Batch rm_scores shape: {rollouts.batch['rm_scores'].shape}")
    print(f"✓ Number of trajectories: {rollouts.batch['input_ids'].shape[0]}")
    
    # Print rewards (similar to agent_proxy.py)
    rm_scores = rollouts.batch["rm_scores"]
    avg_reward = rm_scores.sum(-1).mean().item()
    print(f"🏆 Average reward: {avg_reward:.3f}")
    
    # Print metrics
    if 'metrics' in rollouts.meta_info:
        print("📈 Metrics:")
        for k, v in rollouts.meta_info["metrics"].items():
            print(f"  {k}: {v}")
    
    # Final rollout summary
    print(f"\n🎯 FINAL TEXT ROLLOUT: {rollouts}")
    
    print("✅ Text mode agent proxy test completed successfully!")
    return True

def test_agent_proxy_with_images():
    """Test agent proxy with image mode"""
    print("\n" + "="*60)
    print("🧪 AGENT PROXY IMAGE TEST")
    print("="*60)
    
    # Set up environment for VLLM
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    # Load test configuration
    config_dir = os.path.dirname(os.path.abspath(__file__))
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="test_config.yaml")
    
    # Set to image mode
    config.custom_envs.SimpleSokoban.env_config.render_mode = "rgb_array"
    
    # Use vision model for image mode tests
    config.actor_rollout_ref.model.path = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    print(f"🎮 Environment mode: {config.custom_envs.SimpleSokoban.env_config.render_mode}")
    print(f"🤖 Using vision model: {config.actor_rollout_ref.model.path}")
    
    # Initialize tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    
    try:
        processor = AutoProcessor.from_pretrained(config.actor_rollout_ref.model.path)
        print(f"🤖 Processor loaded for vision support")
    except Exception as e:
        print(f"⚠️  Processor not available: {e}")
        processor = None
    
    # Create EnhancedVllmWrapperWg for real model generation with vision support
    print("🚀 Creating EnhancedVllmWrapperWg for vision model generation...")
    actor_wg = EnhancedVllmWrapperWg(config, tokenizer, processor)
    print("✅ EnhancedVllmWrapperWg created successfully!")
    
    agent_proxy = LLMAgentProxy(config, actor_wg, tokenizer, processor=processor)  # Pass processor for image support
    
    # Create rollout DataProto
    rollout_dataproto = DataProto(
        batch=None, 
        non_tensor_batch=None, 
        meta_info={
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
            'recompute_log_prob': False,
            'do_sample': False,
            'validate': True
        }
    )
    
    # Test the actual rollout method from LLMAgentProxy with images
    print(f"\n🚀 Testing agent_proxy.rollout() method with images...")
    rollouts = agent_proxy.rollout(rollout_dataproto, val=True)
    
    print(f"\n📊 IMAGE ROLLOUT RESULTS:")
    print(f"✓ Successfully processed image-based environments")
    print(f"✓ Generated {rollouts.batch['input_ids'].shape[0]} trajectories")
    
    # Check for multimodal data in final results
    if 'multi_modal_data' in rollouts.non_tensor_batch:
        print("✅ Image mode confirmed - multimodal data present")
    else:
        print("⚠️  No multimodal data in final results")
    
    # Calculate avg reward for final summary
    rm_scores = rollouts.batch["rm_scores"]
    avg_reward = rm_scores.sum(-1).mean().item()
    
    # Final rollout summary
    print(f"\n🎯 FINAL IMAGE ROLLOUT: {rollouts}")
    
    print("✅ Image agent proxy test completed successfully!")
    return True

def main():
    """Run tests for LLMAgentProxy.rollout() method"""
    print("🧪 AGENT PROXY ROLLOUT METHOD TESTS")
    print("="*60)
    
    # Clear Hydra
    GlobalHydra.instance().clear()
    
    try:
        # Run tests with only EnhancedVllmWrapperWg
        test_results = {}
        
        # test_results['text_rollout'] = test_agent_proxy_with_text()
        test_results['image_rollout'] = test_agent_proxy_with_images()
        
        # Summary
        print("\n" + "="*60)
        print("📊 AGENT PROXY ROLLOUT METHOD TEST SUMMARY")
        print("="*60)
        
        all_passed = True
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:20} {status}")
            if not result:
                all_passed = False
        
        if all_passed:
            print("\n🎉 All agent proxy tests passed!")
        else:
            print("\n⚠️  Some tests failed!")
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        GlobalHydra.instance().clear()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
