#!/usr/bin/env python3
"""
SyncMultiTurnRollout Test - Tests rollout logic with mocked LLM responses
"""

import sys
import os
import yaml
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import real components
from transformers import AutoTokenizer
from rollout.sync_multi_turn_rollout import SyncMultiTurnRollout
from tests.rollout_tests.rollout_test_utils import create_mock_actor_wg, create_mock_tokenizer_decode


def setup_logging():
    """Setup logging to stream outputs to test_logs directory"""
    test_logs_dir = os.path.join(os.path.dirname(__file__), 'test_logs')
    os.makedirs(test_logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(test_logs_dir, f"sync_multi_turn_rollout_test_{timestamp}.log")
    
    class Tee:
        def __init__(self, file_path):
            self.file = open(file_path, 'w')
            self.stdout = sys.stdout
        
        def write(self, data):
            self.file.write(data)
            self.file.flush()
            self.stdout.write(data)
        
        def flush(self):
            self.file.flush()
            self.stdout.flush()
        
        def close(self):
            self.file.close()
    
    tee = Tee(log_file)
    sys.stdout = tee
    
    print(f"📝 SyncMultiTurnRollout Test log started at {datetime.now()}")
    print(f"📄 Log file: {log_file}")
    print("=" * 70)
    
    return tee


def load_config():
    """Load configuration from base.yaml and agents.yaml"""
    config_dir = project_root / "configs"
    
    try:
        # Load base config
        with open(config_dir / "base.yaml", 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Load agents config
        with open(config_dir / "agents.yaml", 'r') as f:
            agents_config = yaml.safe_load(f)
        
        # Merge configs
        config = {**base_config, **agents_config}
        
        print(f"✅ Loaded configuration from {config_dir}")
        print(f"   Base config keys: {list(base_config.keys())}")
        print(f"   Agents config keys: {list(agents_config.keys())}")
        
        return config
        
    except FileNotFoundError as e:
        print(f"❌ Could not find config file: {e}")
        raise
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        raise


def create_config_object(config_dict):
    """Create a config object with proper attribute access"""
    class Config:
        def __init__(self, config_dict):
            self._original_dict = config_dict
            for key, value in config_dict.items():
                # Only set attributes for string keys (valid Python identifiers)
                if isinstance(key, str) and key.isidentifier():
                    if isinstance(value, dict):
                        setattr(self, key, Config(value))
                    else:
                        setattr(self, key, value)
        
        def get(self, key, default=None):
            if hasattr(self, key):
                return getattr(self, key)
            return self._original_dict.get(key, default)
        
        def __contains__(self, key):
            return hasattr(self, key) or key in self._original_dict
        
        def __getitem__(self, key):
            if hasattr(self, key):
                attr_value = getattr(self, key)
                # If it's a Config object, return its _original_dict for agent consumption
                if isinstance(attr_value, Config):
                    return attr_value._original_dict
                return attr_value
            return self._original_dict[key]
        
        def __getattr__(self, key):
            if key.startswith('_'):
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
            return self._original_dict.get(key, None)
    
    return Config(config_dict)


def create_real_tokenizer():
    """Create a real tokenizer using Hugging Face"""
    try:
        print("📦 Loading Qwen/Qwen2.5-0.5B tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Apply mock tokenizer decode for testing
        tokenizer = create_mock_tokenizer_decode(tokenizer)
        
        print(f"✅ Real tokenizer loaded successfully")
        print(f"   Vocab size: {tokenizer.vocab_size}")
        print(f"   Pad token: {tokenizer.pad_token}")
        print(f"   EOS token: {tokenizer.eos_token}")
        
        return tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load real tokenizer: {e}")
        raise


def test_sync_multi_turn_rollout_creation():
    """Test SyncMultiTurnRollout creation"""
    print("🔍 Testing SyncMultiTurnRollout creation...")
    
    # Load real configuration
    config = load_config()
    config_obj = create_config_object(config)
    
    # Debug: Print detailed config structure
    print(f"   📊 DETAILED CONFIG DEBUG INFO:")
    print(f"      Config type: {type(config_obj)}")
    print(f"      Config top-level keys: {list(config_obj._original_dict.keys()) if hasattr(config_obj, '_original_dict') else 'NO_DICT'}")
    
    # Check required config values for sync_multi_turn_rollout
    required_configs = {
        'max_prompt_length': 'cfg.max_prompt_length',
        'rollout.truncation': 'cfg.rollout.truncation',
        'rollout.agent_group_num': 'cfg.rollout.agent_group_num',
        'rollout.agent_group_size': 'cfg.rollout.agent_group_size',
        'rollout.train': 'cfg.rollout.train',
        'rollout.enable_think': 'cfg.rollout.enable_think',
        'rollout.use_turn_scores': 'cfg.rollout.use_turn_scores',
        'rollout.enable_response_mask': 'cfg.rollout.enable_response_mask',
    }
    
    print(f"\n      📋 REQUIRED CONFIG CHECK:")
    for config_path, access_path in required_configs.items():
        try:
            if '.' in config_path:
                # Handle nested config access
                parts = config_path.split('.')
                current_obj = config_obj
                for part in parts:
                    current_obj = getattr(current_obj, part)
                value = current_obj
                status = "✅ FOUND"
            else:
                # Handle top-level config access
                value = getattr(config_obj, config_path)
                status = "✅ FOUND"
        except AttributeError as e:
            value = f"❌ MISSING: {str(e)}"
            status = "❌ MISSING"
        
        print(f"         {access_path}: {value} ({status})")
    
    # Check rollout config in detail
    print(f"\n      📋 ROLLOUT CONFIG DETAILS:")
    if hasattr(config_obj, 'rollout'):
        rollout_config = getattr(config_obj, 'rollout')
        print(f"         Type: {type(rollout_config)}")
        print(f"         Keys: {list(rollout_config._original_dict.keys()) if hasattr(rollout_config, '_original_dict') else 'NO_DICT'}")
        
        # Check each rollout attribute
        rollout_attrs = ['agent_group_num', 'agent_group_size', 'train', 'validation', 'truncation', 'enable_think', 'use_turn_scores', 'enable_response_mask']
        for attr in rollout_attrs:
            try:
                value = getattr(rollout_config, attr)
                print(f"         {attr}: {value} (✅ FOUND)")
            except AttributeError:
                print(f"         {attr}: ❌ MISSING")
    else:
        print(f"         ❌ NO ROLLOUT CONFIG FOUND!")
    
    # Check sokobanAgent config
    print(f"\n      📋 SOKOBAN AGENT CONFIG:")
    if hasattr(config_obj, 'sokobanAgent'):
        sokoban_config = getattr(config_obj, 'sokobanAgent')
        print(f"         Type: {type(sokoban_config)}")
        print(f"         Keys: {list(sokoban_config._original_dict.keys()) if hasattr(sokoban_config, '_original_dict') else 'NO_DICT'}")
        
        # Check agent_config nested structure
        if hasattr(sokoban_config, 'agent_config'):
            agent_config = sokoban_config.agent_config
            print(f"         agent_config type: {type(agent_config)}")
            print(f"         agent_config keys: {list(agent_config._original_dict.keys()) if hasattr(agent_config, '_original_dict') else 'NO_DICT'}")
            
            # Check max_turns specifically
            try:
                max_turns = agent_config.max_turns
                print(f"         max_turns: {max_turns} (✅ FOUND)")
            except AttributeError:
                print(f"         max_turns: ❌ MISSING")
        else:
            print(f"         ❌ NO AGENT_CONFIG FOUND!")
    else:
        print(f"         ❌ NO SOKOBAN AGENT CONFIG FOUND!")
    
    # Show what we have vs what we need
    print(f"\n      🔍 CONFIG AVAILABILITY SUMMARY:")
    print(f"         Available top-level keys: {list(config_obj._original_dict.keys()) if hasattr(config_obj, '_original_dict') else 'NONE'}")
    print(f"         Need for rollout: max_prompt_length, rollout.truncation, rollout.agent_group_num, rollout.agent_group_size, etc.")
    print(f"         Need for agents: sokobanAgent.agent_config.max_turns")
    
    # Create real tokenizer
    tokenizer = create_real_tokenizer()
    
    # Create mock actor worker group (only thing we mock)
    mock_actor_wg = create_mock_actor_wg()
    
    # Create rollout manager
    rollout = SyncMultiTurnRollout(
        actor_rollout_wg=mock_actor_wg,
        cfg=config_obj,
        tokenizer=tokenizer,
        processor=None  # Not used in our tests
    )
    
    # Verify creation - now use rollout config structure
    assert rollout.cfg == config_obj
    assert rollout.tokenizer == tokenizer
    assert rollout.actor_wg == mock_actor_wg
    
    # Calculate expected number of agents from agent_group_num * agent_group_size
    expected_n_agents = config['rollout']['agent_group_num'] * config['rollout']['agent_group_size']
    assert rollout.n_agents == expected_n_agents
    assert len(rollout.agents) == expected_n_agents
    assert rollout.step_cnt == 0
    
    print(f"✅ SyncMultiTurnRollout created successfully")
    print(f"   Number of agents: {rollout.n_agents} (calculated as {config['rollout']['agent_group_num']} groups × {config['rollout']['agent_group_size']} agents/group)")
    print(f"   Agent class: {rollout.agent_cls}")
    print(f"   Max turns: {rollout.max_turns}")
    print(f"   Step count: {rollout.step_cnt}")
    
    rollout.close()


def test_sync_multi_turn_rollout_full_rollout():
    """Test complete rollout process"""
    print("🔍 Testing complete rollout process...")
    
    config = load_config()
    config_obj = create_config_object(config)
    tokenizer = create_real_tokenizer()
    mock_actor_wg = create_mock_actor_wg()
    
    rollout = SyncMultiTurnRollout(
        actor_rollout_wg=mock_actor_wg,
        cfg=config_obj,
        tokenizer=tokenizer,
        processor=None
    )
    
    print(f"\n   🚀 STARTING FULL ROLLOUT")
    print("   " + "=" * 50)
    
    # Execute rollout
    final_env_outs = rollout.rollout()
    
    # Verify rollout results
    assert len(final_env_outs) == rollout.n_agents
    assert rollout.step_cnt > 0
    
    print(f"\n   📊 ROLLOUT RESULTS:")
    print(f"      Total steps: {rollout.step_cnt}")
    print(f"      Number of agents: {len(final_env_outs)}")
    
    for i, env_out in enumerate(final_env_outs):
        print(f"      Agent {i}: Done={env_out.done}, Reward={env_out.reward}")
        
        # Check agent trajectory
        agent = rollout.agents[i]
        print(f"         Trajectory length: {len(agent.trajectory_history)}")
        print(f"         Total actions consumed: {agent.total_actions_consumed}")
        print(f"         Current turn: {agent.cur_turn}")
    
    print(f"\n   ✅ Full rollout completed successfully")
    
    rollout.close()


def test_sync_multi_turn_rollout_final_states():
    """Test final rollout states collection"""
    print("🔍 Testing final rollout states collection...")
    
    config = load_config()
    config_obj = create_config_object(config)
    tokenizer = create_real_tokenizer()
    mock_actor_wg = create_mock_actor_wg()
    
    rollout = SyncMultiTurnRollout(
        actor_rollout_wg=mock_actor_wg,
        cfg=config_obj,
        tokenizer=tokenizer,
        processor=None
    )
    
    # Execute rollout
    final_env_outs = rollout.rollout()
    
    # Collect final rollout states
    final_rollout_states = rollout._collect_final_rollout_states()
    
    # Verify final states
    assert len(final_rollout_states) == rollout.n_agents
    
    print(f"\n   📊 FINAL ROLLOUT STATES:")
    print(f"      Number of agent states: {len(final_rollout_states)}")
    
    for i, state in enumerate(final_rollout_states):
        print(f"\n      Agent {i} Final State:")
        print(f"         Environment ID: {state['env_id']}")
        print(f"         Group ID: {state['group_id']}")
        print(f"         Tag: {state['tag']}")
        print(f"         History Length: {len(state['history'])}")
        print(f"         Penalty: {state.get('penalty', 0.0)}")
        
        # Print metrics
        print(f"         Metrics Values:")
        for key, value in state['metrics'].items():
            print(f"            {key}: {value}")
        
        # Print trajectory history
        print(f"         Trajectory History:")
        for j, step in enumerate(state['history']):
            print(f"            Step {j+1}:")
            print(f"               Actions Left: {step['actions_left']}")
            print(f"               Actions: {step['actions']}")
            print(f"               Reward: {step['reward']}")
            print(f"               Info: {step['info']}")
            print(f"               LLM Response: {step['llm_response']}")
            
            # Print state (first few lines)
            state_lines = step['state'].split('\n')[:3]
            print(f"               State (first 3 lines):")
            for line in state_lines:
                print(f"                  {line}")
    
    print(f"\n   ✅ Final rollout states collected successfully")
    
    rollout.close()


def test_sync_multi_turn_rollout_ppo_batch():
    """Test PPO batch building"""
    print("🔍 Testing PPO batch building...")
    
    config = load_config()
    config_obj = create_config_object(config)
    tokenizer = create_real_tokenizer()
    mock_actor_wg = create_mock_actor_wg()
    
    rollout = SyncMultiTurnRollout(
        actor_rollout_wg=mock_actor_wg,
        cfg=config_obj,
        tokenizer=tokenizer,
        processor=None
    )
    
    # Execute rollout
    final_env_outs = rollout.rollout()
    
    # Build PPO batch
    ppo_batch = rollout.build_ppo_batch()
    
    # Verify PPO batch
    assert ppo_batch is not None
    assert hasattr(ppo_batch, 'batch')
    assert hasattr(ppo_batch, 'non_tensor_batch')
    assert hasattr(ppo_batch, 'meta_info')
    
    # Check batch tensors
    batch_keys = ['input_ids', 'attention_mask', 'position_ids', 'responses', 'loss_mask', 'rm_scores', 'original_rm_scores']
    for key in batch_keys:
        assert key in ppo_batch.batch, f"Missing key: {key}"
    
    # Check non-tensor batch
    non_tensor_keys = ['env_ids', 'group_ids', 'messages_list']
    for key in non_tensor_keys:
        assert key in ppo_batch.non_tensor_batch, f"Missing non-tensor key: {key}"
    
    # Check meta info
    assert 'metrics' in ppo_batch.meta_info
    
    print(f"\n   📊 PPO BATCH STRUCTURE:")
    print(f"      Batch keys: {list(ppo_batch.batch.keys())}")
    print(f"      Non-tensor batch keys: {list(ppo_batch.non_tensor_batch.keys())}")
    print(f"      Meta info keys: {list(ppo_batch.meta_info.keys())}")
    
    # Print tensor shapes
    print(f"\n      Tensor Shapes:")
    for key, tensor in ppo_batch.batch.items():
        if hasattr(tensor, 'shape'):
            print(f"         {key}: {tensor.shape}")
        else:
            print(f"         {key}: {type(tensor)}")
    
    # Print metrics
    print(f"\n      Metrics:")
    for key, value in ppo_batch.meta_info['metrics'].items():
        print(f"         {key}: {value}")
    
    # Add detailed DataProto visualization
    print(f"\n   📊 DETAILED DATAPROTO VISUALIZATION:")
    print("   " + "=" * 50)
    
    # Check length
    try:
        print(f"      DataProto length: {len(ppo_batch)}")
    except:
        print(f"      DataProto length: Not available")
    
    # Access tensor data
    print(f"\n      Tensor Data (ppo_batch.batch):")
    print(f"         Type: {type(ppo_batch.batch)}")
    if hasattr(ppo_batch.batch, 'batch_size'):
        print(f"         Batch size: {ppo_batch.batch.batch_size}")
    
    for key, tensor in ppo_batch.batch.items():
        if hasattr(tensor, 'shape'):
            print(f"         {key}:")
            print(f"            Shape: {tensor.shape}")
            print(f"            Dtype: {tensor.dtype}")
            print(f"            Device: {tensor.device}")
            if tensor.numel() > 0:
                print(f"            Min/Max: {tensor.min().item():.4f} / {tensor.max().item():.4f}")
                print(f"            Sample values: {tensor.flatten()[:5].tolist()}")
        else:
            print(f"         {key}: {type(tensor)} - {tensor}")
    
    # Access non-tensor data
    print(f"\n      Non-tensor Data (ppo_batch.non_tensor_batch):")
    print(f"         Type: {type(ppo_batch.non_tensor_batch)}")
    for key, value in ppo_batch.non_tensor_batch.items():
        print(f"         {key}:")
        print(f"            Type: {type(value)}")
        if hasattr(value, 'shape'):
            print(f"            Shape: {value.shape}")
        if hasattr(value, '__len__'):
            print(f"            Length: {len(value)}")
        if hasattr(value, 'dtype'):
            print(f"            Dtype: {value.dtype}")
        
        # Print sample values
        if key == 'env_ids':
            print(f"            Sample values: {value[:3] if len(value) > 3 else value}")
        elif key == 'group_ids':
            print(f"            Sample values: {value[:3] if len(value) > 3 else value}")
        elif key == 'messages_list':
            print(f"            Sample (first message):")
            if len(value) > 0 and len(value[0]) > 0:
                first_msg = value[0][0]
                print(f"               Role: {first_msg.get('role', 'N/A')}")
                content_preview = first_msg.get('content', '')[:100] + '...' if len(first_msg.get('content', '')) > 100 else first_msg.get('content', '')
                print(f"               Content: {content_preview}")
    
    # Access metadata
    print(f"\n      Metadata (ppo_batch.meta_info):")
    print(f"         Type: {type(ppo_batch.meta_info)}")
    for key, value in ppo_batch.meta_info.items():
        print(f"         {key}:")
        print(f"            Type: {type(value)}")
        if isinstance(value, dict):
            print(f"            Keys: {list(value.keys())}")
            for sub_key, sub_value in value.items():
                print(f"               {sub_key}: {sub_value} ({type(sub_value)})")
        else:
            print(f"            Value: {value}")
    
    print(f"\n   ✅ PPO batch built successfully")
    
    rollout.close()


def test_sync_multi_turn_rollout_complete():
    """Test complete rollout workflow with detailed output"""
    print("🔍 Testing complete rollout workflow...")
    
    config = load_config()
    config_obj = create_config_object(config)
    tokenizer = create_real_tokenizer()
    mock_actor_wg = create_mock_actor_wg()
    
    rollout = SyncMultiTurnRollout(
        actor_rollout_wg=mock_actor_wg,
        cfg=config_obj,
        tokenizer=tokenizer,
        processor=None
    )
    
    print(f"\n   🚀 COMPLETE ROLLOUT WORKFLOW")
    print("   " + "=" * 60)
    
    # Step 1: Execute rollout
    print(f"\n   Step 1: Executing rollout...")
    final_env_outs = rollout.rollout()
    print(f"   ✅ Rollout completed with {rollout.step_cnt} steps")
    
    # Step 2: Collect final rollout states
    print(f"\n   Step 2: Collecting final rollout states...")
    final_rollout_states = rollout._collect_final_rollout_states()
    print(f"   ✅ Collected states from {len(final_rollout_states)} agents")
    
    # Step 3: Build PPO batch
    print(f"\n   Step 3: Building PPO batch...")
    ppo_batch = rollout.build_ppo_batch()
    print(f"   ✅ PPO batch built successfully")
    
    # Print detailed final rollout states
    print(f"\n   📊 DETAILED FINAL ROLLOUT STATES:")
    print("   " + "=" * 60)
    
    for i, state in enumerate(final_rollout_states):
        print(f"\n   Agent {i} Complete State:")
        print(f"      Environment ID: {state['env_id']}")
        print(f"      Group ID: {state['group_id']}")
        print(f"      Tag: {state['tag']}")
        print(f"      History Length: {len(state['history'])}")
        print(f"      Penalty: {state.get('penalty', 0.0)}")
        
        print(f"\n      Metrics:")
        for key, value in state['metrics'].items():
            print(f"         {key}: {value}")
        
        print(f"\n      Complete Trajectory History:")
        for j, step in enumerate(state['history']):
            print(f"\n         Turn {j+1}:")
            print(f"            Actions Left: {step['actions_left']}")
            print(f"            Actions Executed: {step['actions']}")
            print(f"            Reward: {step['reward']}")
            print(f"            LLM Response: {step['llm_response']}")
            print(f"            LLM Raw Response: {step['llm_raw_response']}")
            
            # Print step info
            print(f"            Step Info:")
            for info_key, info_value in step['info'].items():
                print(f"               {info_key}: {info_value}")
            
            # Print environment state
            print(f"            Environment State:")
            state_lines = step['state'].split('\n')
            for line in state_lines:
                print(f"               {line}")
    
    # Print PPO batch summary
    print(f"\n   📊 PPO BATCH SUMMARY:")
    print("   " + "=" * 60)
    print(f"      Batch size: {len(final_rollout_states)}")
    print(f"      Total metrics: {len(ppo_batch.meta_info['metrics'])}")
    
    print(f"\n      Final Metrics Summary:")
    for key, value in ppo_batch.meta_info['metrics'].items():
        print(f"         {key}: {value}")
    
    print(f"\n   ✅ Complete rollout workflow test passed")
    
    rollout.close()


if __name__ == "__main__":
    # Setup logging to test_logs
    tee = setup_logging()
    
    try:
        print("🚀 Starting SyncMultiTurnRollout Tests...")
        print()
        
        print("Test 1: Rollout creation")
        test_sync_multi_turn_rollout_creation()
        print()
        
        print("Test 2: Full rollout process")
        test_sync_multi_turn_rollout_full_rollout()
        print()
        
        print("Test 3: Final rollout states collection")
        test_sync_multi_turn_rollout_final_states()
        print()
        
        print("Test 4: PPO batch building")
        test_sync_multi_turn_rollout_ppo_batch()
        print()
        
        print("Test 5: Complete rollout workflow")
        test_sync_multi_turn_rollout_complete()
        print()
        
        print("=" * 70)
        print("🎉 All SyncMultiTurnRollout tests passed!")
        print(f"✅ Test completed at {datetime.now()}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Close the log file
        tee.close()
        sys.stdout = tee.stdout  # Restore original stdout
