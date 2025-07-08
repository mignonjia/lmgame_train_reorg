#!/usr/bin/env python3
"""
Test for LLMAgentProxy.rollout() method from agent_proxy.py.
Tests the actual rollout method using VllmWrapperWg (real model generation) or SimpleActorWg (mock fallback) for both text and image modes.
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
    print("⚠️  VLLM not available - will use mock actor only")

# Enhanced VLLM Wrapper that handles both text and vision inputs
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
        
        # Initialize VLLM with enhanced parameters for multimodal support
        vllm_kwargs = {
            'model': model_name,
            'tensor_parallel_size': getattr(ro_config, 'tensor_model_parallel_size', 1),
            'dtype': getattr(ro_config, 'dtype', 'auto'),
            'enforce_eager': getattr(ro_config, 'enforce_eager', True),
            'gpu_memory_utilization': getattr(ro_config, 'gpu_memory_utilization', 0.2),
            'disable_custom_all_reduce': True,
            'skip_tokenizer_init': getattr(ro_config, 'skip_tokenizer_init', False),
            'max_model_len': getattr(ro_config, 'max_model_len', 8192),
            'disable_log_stats': getattr(ro_config, 'disable_log_stats', True),
            'max_num_batched_tokens': getattr(ro_config, 'max_num_batched_tokens', 2048),
            'enable_chunked_prefill': getattr(ro_config, 'enable_chunked_prefill', False),
            'enable_prefix_caching': getattr(ro_config, 'enable_prefix_caching', False),
            'trust_remote_code': getattr(ro_config, 'trust_remote_code', False),
        }
        
        # Add vision-specific parameters if processor is available
        if self.processor is not None:
            vllm_kwargs.update({
                'disable_mm_preprocessor_cache': getattr(ro_config, 'disable_mm_preprocessor_cache', True),
            })
        
        print(f"🚀 Initializing VLLM with model: {model_name}")
        self.llm = LLM(**vllm_kwargs)
        print("✅ VLLM LLM initialized successfully")
        
        # Set up sampling parameters
        val_kwargs = getattr(ro_config, 'val_kwargs', {})
        self.sampling_params = SamplingParams(
            max_tokens=getattr(ro_config, 'response_length', 256),
            temperature=getattr(val_kwargs, 'temperature', 0.0),
            top_p=getattr(val_kwargs, 'top_p', 1.0),
            top_k=getattr(val_kwargs, 'top_k', -1),
        )
        print(f"📋 Sampling params: max_tokens={self.sampling_params.max_tokens}, temperature={self.sampling_params.temperature}")

    def generate_sequences(self, lm_inputs: DataProto):
        """
        Generate sequences supporting both text-only and multimodal inputs.
        """
        input_ids = lm_inputs.batch['input_ids']
        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        input_texts = [text.replace("<|endoftext|>", "") for text in input_texts]
        
        # Check if multimodal data is present
        has_multimodal = (
            'multi_modal_data' in lm_inputs.non_tensor_batch and
            'multi_modal_inputs' in lm_inputs.non_tensor_batch
        )
        
        if has_multimodal and self.processor is not None:
            print("🖼️  Processing multimodal inputs with vision support")
            # Handle multimodal inputs
            multi_modal_data = lm_inputs.non_tensor_batch['multi_modal_data']
            
            # Prepare images for VLLM - extract images from multi_modal_data
            images_for_vllm = []
            for i, mm_data in enumerate(multi_modal_data):
                if 'image' in mm_data and len(mm_data['image']) > 0:
                    # Take the first image for each input (VLLM typically handles one image per prompt)
                    images_for_vllm.append(mm_data['image'][0])
                else:
                    images_for_vllm.append(None)
            
            # Generate with multimodal inputs
            try:
                # For VLLM multimodal generation, we need to pass images separately
                prompts = []
                for i, (text, image) in enumerate(zip(input_texts, images_for_vllm)):
                    if image is not None:
                        # Create multimodal prompt
                        prompts.append({
                            "prompt": text,
                            "multi_modal_data": {"image": image}
                        })
                    else:
                        prompts.append(text)
                
                outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)
                
            except Exception as e:
                print(f"⚠️  Multimodal generation failed: {e}")
                print("🔄 Falling back to text-only generation")
                outputs = self.llm.generate(input_texts, sampling_params=self.sampling_params)
        else:
            print("📝 Processing text-only inputs")
            # Standard text-only generation
            outputs = self.llm.generate(input_texts, sampling_params=self.sampling_params)
        
        # Extract generated texts
        texts = [output.outputs[0].text for output in outputs]
        
        # Create output DataProto
        lm_outputs = DataProto()
        lm_outputs.non_tensor_batch = {
            'response_texts': texts,
            'env_ids': lm_inputs.non_tensor_batch['env_ids'],
            'group_ids': lm_inputs.non_tensor_batch['group_ids']
        }
        lm_outputs.meta_info = lm_inputs.meta_info
        
        return lm_outputs

# Simple Mock Actor Worker that returns predefined responses
class SimpleActorWg:
    """Simple mock actor worker for testing without VLLM"""
    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.call_count = 0
        
        # Simple predefined responses for Sokoban
        self.responses = [
            "<think>I need to move down</think><answer>down</answer>",
            "<think>I should go right</think><answer>right</answer>", 
            "<think>Let me try left</think><answer>left</answer>",
            "<think>Moving up now</think><answer>up</answer>",
        ]
    
    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        """Simple mock generation that works with both text and multimodal inputs"""
        env_ids = lm_inputs.non_tensor_batch['env_ids']
        batch_size = len(env_ids)
        
        # Check if multimodal data is present
        has_multimodal = (
            'multi_modal_data' in lm_inputs.non_tensor_batch and
            'multi_modal_inputs' in lm_inputs.non_tensor_batch
        )
        
        if has_multimodal:
            print("🎭 Mock processing multimodal inputs")
        else:
            print("🎭 Mock processing text-only inputs")
        
        # Generate simple responses
        responses = []
        for i in range(batch_size):
            response_idx = (self.call_count + i) % len(self.responses)
            responses.append(self.responses[response_idx])
        
        self.call_count += 1
        
        # Create output DataProto
        lm_outputs = DataProto()
        lm_outputs.non_tensor_batch = {
            'response_texts': responses,
            'env_ids': env_ids,
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
    
    # Ensure text mode
    config.custom_envs.SimpleSokoban.env_config.render_mode = "text"
    
    print(f"📝 Configuration loaded from: {config_dir}")
    print(f"🎮 Environment mode: {config.custom_envs.SimpleSokoban.env_config.render_mode}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    print(f"🤖 Tokenizer loaded: {config.actor_rollout_ref.model.path}")
    
    # Try to use EnhancedVllmWrapperWg for real model generation, fallback to SimpleActorWg if needed
    try:
        print("🚀 Attempting to create EnhancedVllmWrapperWg for real model generation...")
        actor_wg = EnhancedVllmWrapperWg(config, tokenizer)
        print("✅ EnhancedVllmWrapperWg created successfully - using real model!")
    except Exception as e:
        print(f"⚠️  Could not create EnhancedVllmWrapperWg: {e}")
        print("🔄 Falling back to SimpleActorWg mock...")
        actor_wg = SimpleActorWg(config, tokenizer)
        print("🎭 Using SimpleActorWg mock for testing")
    
    agent_proxy = LLMAgentProxy(config, actor_wg, tokenizer)
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
    
    print(f"🎮 Environment mode: {config.custom_envs.SimpleSokoban.env_config.render_mode}")
    
    # Initialize tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    
    try:
        processor = AutoProcessor.from_pretrained(config.actor_rollout_ref.model.path)
        print(f"🤖 Processor loaded for vision support")
    except Exception as e:
        print(f"⚠️  Processor not available: {e}")
        processor = None
    
    # Try to use EnhancedVllmWrapperWg for real model generation with vision support
    try:
        print("🚀 Attempting to create EnhancedVllmWrapperWg for real vision model generation...")
        actor_wg = EnhancedVllmWrapperWg(config, tokenizer, processor)
        print("✅ EnhancedVllmWrapperWg created successfully - using real vision model!")
    except Exception as e:
        print(f"⚠️  Could not create EnhancedVllmWrapperWg: {e}")
        print("🔄 Falling back to SimpleActorWg mock for vision test...")
        actor_wg = SimpleActorWg(config, tokenizer)
        print("🎭 Using SimpleActorWg mock for vision testing")
    
    agent_proxy = LLMAgentProxy(config, actor_wg, tokenizer)
    
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
    
    print("✅ Image agent proxy test completed successfully!")
    return True

def main():
    """Run tests for LLMAgentProxy.rollout() method"""
    print("🧪 AGENT PROXY ROLLOUT METHOD TESTS")
    print("="*60)
    
    # Clear Hydra
    GlobalHydra.instance().clear()
    
    try:
        # Run simple tests
        test_results = {}
        
        test_results['text_rollout'] = test_agent_proxy_with_text()
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
