import pytest
import torch
import numpy as np
from unittest.mock import Mock
import sys
import os
from datetime import datetime
from transformers import AutoTokenizer
from tensordict import TensorDict

# Add paths to import VERL components
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../verl'))

from verl import DataProto
from rollout.sync_multi_turn_rollout import SyncMultiTurnRollout


# Setup logging to file
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
    
    print(f"📝 Test log started at {datetime.now()}")
    print(f"📄 Log file: {log_file}")
    print("=" * 50)
    
    return tee


class MockAgent:
    """Simple mock agent that returns prompt strings for the new interface"""
    
    def __init__(self, agent_id, tokenizer):
        self.agent_id = agent_id
        self.tokenizer = tokenizer
    
    def get_llm_prompts(self, env_out):
        """Return a single prompt string (new interface)"""
        return f"Agent {self.agent_id}: What is 2+2? Please solve this math problem."
    
    def get_initial_env_outputs(self):
        return Mock(done=False)


class MockConfig:
    """Simple mock config"""
    def __init__(self):
        self.agent_batch_size = 2
        self.n_gpus_per_node = 1
        self.train = ["sokobanAgent"]
        self.max_prompt_length = 128
        self.truncation = "right"


def test_collect_prompts_basic():
    """Test _collect_prompts converts string prompts to DataProto batch correctly"""
    
    # Setup Qwen tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup
    config = MockConfig()
    mock_actor_wg = Mock()
    mock_processor = Mock()
    
    # Create rollout instance
    rollout = SyncMultiTurnRollout(
        actor_rollout_wg=mock_actor_wg,
        cfg=config,
        tokenizer=tokenizer,
        processor=mock_processor
    )
    
    # Replace with mock agents
    rollout.agents = [MockAgent(0, tokenizer), MockAgent(1, tokenizer)]
    rollout.n_agents = 2
    rollout.done_mask = torch.zeros(2, dtype=torch.bool)
    rollout.env_outs = [agent.get_initial_env_outputs() for agent in rollout.agents]
    
    print("🔍 Testing basic _collect_prompts functionality...")
    print("Agent 0 prompt:", rollout.agents[0].get_llm_prompts(rollout.env_outs[0]))
    print("Agent 1 prompt:", rollout.agents[1].get_llm_prompts(rollout.env_outs[1]))
    print()
    
    # Test _collect_prompts
    batched_dataproto, idx_map = rollout._collect_prompts()
    
    # Verify results
    assert batched_dataproto is not None
    assert idx_map == [0, 1]
    
    # Check batch dimensions - all three required elements
    assert batched_dataproto.batch["input_ids"].shape[0] == 2  # 2 agents
    assert batched_dataproto.batch["attention_mask"].shape[0] == 2
    assert batched_dataproto.batch["position_ids"].shape[0] == 2
    
    print(f"✅ Batch DataProto created successfully:")
    print(f"   - input_ids shape: {batched_dataproto.batch['input_ids'].shape}")
    print(f"   - attention_mask shape: {batched_dataproto.batch['attention_mask'].shape}")
    print(f"   - position_ids shape: {batched_dataproto.batch['position_ids'].shape}")
    print(f"   - idx_map: {idx_map}")
    print()
    
    # Test roundtrip: Decode the tokens back to verify content preservation
    batch_input_ids = batched_dataproto.batch["input_ids"]
    batch_attention_mask = batched_dataproto.batch["attention_mask"]
    
    print("🔄 Testing roundtrip: tokens → text")
    for i in range(2):
        # Extract valid tokens (non-padding)
        valid_mask = batch_attention_mask[i] == 1
        valid_tokens = batch_input_ids[i][valid_mask]
        
        reconstructed_text = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()
        original_prompt = rollout.agents[i].get_llm_prompts(rollout.env_outs[i])
        
        print(f"   Agent {i}:")
        print(f"     Original: {original_prompt}")
        print(f"     Reconstructed: {reconstructed_text}")
        
        # Verify core content is preserved
        assert f"Agent {i}" in reconstructed_text
        assert "What is 2+2?" in reconstructed_text
    
    print("✅ Basic collect_prompts test passed")


def test_collect_prompts_two_step_process():
    """Test the two-step process: _collect_prompts_from_agents + _convert_prompts_to_dataproto"""
    
    # Setup Qwen tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup
    config = MockConfig()
    rollout = SyncMultiTurnRollout(
        actor_rollout_wg=Mock(),
        cfg=config,
        tokenizer=tokenizer,
        processor=Mock()
    )
    
    # Replace with mock agents
    rollout.agents = [MockAgent(0, tokenizer), MockAgent(1, tokenizer)]
    rollout.n_agents = 2
    rollout.done_mask = torch.zeros(2, dtype=torch.bool)
    rollout.env_outs = [agent.get_initial_env_outputs() for agent in rollout.agents]
    
    print("🔍 Testing two-step process...")
    
    # Step 1: Test _collect_prompts_from_agents
    prompts, idx_map = rollout._collect_prompts_from_agents()
    
    assert prompts is not None
    assert len(prompts) == 2
    assert idx_map == [0, 1]
    assert "Agent 0" in prompts[0]
    assert "Agent 1" in prompts[1]
    
    print(f"✅ Step 1 - Collected prompts:")
    if prompts is not None and idx_map is not None:
        for i, prompt in enumerate(prompts):
            print(f"   [{idx_map[i]}]: {prompt}")
    print()
    
    # Step 2: Test _convert_prompts_to_dataproto
    if prompts is not None:
        batched_dataproto = rollout._convert_prompts_to_dataproto(prompts)
        
        assert batched_dataproto is not None
        assert batched_dataproto.batch["input_ids"].shape[0] == 2
        assert batched_dataproto.batch["attention_mask"].shape[0] == 2
        assert batched_dataproto.batch["position_ids"].shape[0] == 2
        
        print(f"✅ Step 2 - Converted to DataProto:")
        print(f"   - input_ids shape: {batched_dataproto.batch['input_ids'].shape}")
        print(f"   - attention_mask shape: {batched_dataproto.batch['attention_mask'].shape}")
        print(f"   - position_ids shape: {batched_dataproto.batch['position_ids'].shape}")
        print()
        
        # Step 3: Verify end-to-end matches _collect_prompts
        final_dataproto, final_idx_map = rollout._collect_prompts()
        
        assert final_dataproto is not None
        assert final_idx_map is not None
        assert torch.equal(batched_dataproto.batch["input_ids"], final_dataproto.batch["input_ids"])
        assert torch.equal(batched_dataproto.batch["attention_mask"], final_dataproto.batch["attention_mask"])
        assert torch.equal(batched_dataproto.batch["position_ids"], final_dataproto.batch["position_ids"])
        assert idx_map == final_idx_map
    
    print("✅ Two-step process test passed - matches end-to-end result")


def test_collect_prompts_with_done_agent():
    """Test _collect_prompts skips done agents"""
    
    # Setup Qwen tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup
    config = MockConfig()
    rollout = SyncMultiTurnRollout(
        actor_rollout_wg=Mock(),
        cfg=config,
        tokenizer=tokenizer,
        processor=Mock()
    )
    
    # Replace with mock agents
    rollout.agents = [MockAgent(0, tokenizer), MockAgent(1, tokenizer)]
    rollout.n_agents = 2
    rollout.done_mask = torch.tensor([True, False])  # First agent done
    rollout.env_outs = [agent.get_initial_env_outputs() for agent in rollout.agents]
    
    print("🔍 Testing with done agent (agent 0 done, agent 1 active)...")
    
    # Test _collect_prompts
    batched_dataproto, idx_map = rollout._collect_prompts()
    
    # Should only have second agent
    assert batched_dataproto is not None
    assert idx_map is not None
    assert idx_map == [1]
    assert batched_dataproto.batch["input_ids"].shape[0] == 1
    
    # Verify it's agent 1's prompt
    reconstructed_text = tokenizer.decode(
        batched_dataproto.batch["input_ids"][0], 
        skip_special_tokens=True
    ).strip()
    assert "Agent 1" in reconstructed_text
    
    print(f"✅ Done agent test passed:")
    print(f"   - Only active agent included: {idx_map}")
    print(f"   - Batch size: {batched_dataproto.batch['input_ids'].shape[0]}")
    print(f"   - Content: {reconstructed_text}")


def test_collect_prompts_all_done():
    """Test _collect_prompts returns None when all agents done"""
    
    # Setup Qwen tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup
    config = MockConfig()
    rollout = SyncMultiTurnRollout(
        actor_rollout_wg=Mock(),
        cfg=config,
        tokenizer=tokenizer,
        processor=Mock()
    )
    
    # Replace with mock agents  
    rollout.agents = [MockAgent(0, tokenizer), MockAgent(1, tokenizer)]
    rollout.n_agents = 2
    rollout.done_mask = torch.tensor([True, True])  # All done
    rollout.env_outs = [agent.get_initial_env_outputs() for agent in rollout.agents]
    
    print("🔍 Testing all done scenario...")
    
    # Test _collect_prompts
    result = rollout._collect_prompts()
    
    # Should return None, None
    assert result == (None, None)
    
    print("✅ All done test passed - returns (None, None)")


def test_collect_prompts_gpu_padding():
    """Test GPU padding when batch size not divisible by n_gpus_per_node"""
    
    # Setup Qwen tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup with 3 agents, 2 GPUs per node (should pad to 4)
    config = MockConfig()
    config.agent_batch_size = 3
    config.n_gpus_per_node = 2
    
    rollout = SyncMultiTurnRollout(
        actor_rollout_wg=Mock(),
        cfg=config,
        tokenizer=tokenizer,
        processor=Mock()
    )
    
    # Replace with mock agents
    rollout.agents = [MockAgent(0, tokenizer), MockAgent(1, tokenizer), MockAgent(2, tokenizer)]
    rollout.n_agents = 3
    rollout.done_mask = torch.zeros(3, dtype=torch.bool)
    rollout.env_outs = [agent.get_initial_env_outputs() for agent in rollout.agents]
    
    print("🔍 Testing GPU padding (3 agents, 2 GPUs → should pad to 4)...")
    
    # Test _collect_prompts
    batched_dataproto, idx_map = rollout._collect_prompts()
    
    # Should pad to 4 (next multiple of 2)
    assert batched_dataproto is not None
    assert idx_map is not None
    assert batched_dataproto.batch["input_ids"].shape[0] == 4
    assert len(idx_map) == 3  # idx_map doesn't include padding
    
    print(f"✅ GPU padding test passed:")
    print(f"   - Original agents: 3")
    print(f"   - Batch size after padding: {batched_dataproto.batch['input_ids'].shape[0]}")
    print(f"   - idx_map length: {len(idx_map)}")


if __name__ == "__main__":
    # Setup logging to test_logs
    tee = setup_logging()
    
    try:
        print("🚀 Starting _collect_prompts tests with new string interface...")
        print()
        
        print("Test 1: Basic functionality")
        test_collect_prompts_basic()
        print()
        
        print("Test 2: Two-step process verification")
        test_collect_prompts_two_step_process()
        print()
        
        print("Test 3: Done agent handling")
        test_collect_prompts_with_done_agent()
        print()
        
        print("Test 4: All done scenario")
        test_collect_prompts_all_done()
        print()
        
        print("Test 5: GPU padding")
        test_collect_prompts_gpu_padding()
        print()
        
        print("=" * 50)
        print("🎉 All tests passed!")
        print(f"✅ Test completed at {datetime.now()}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Close the log file
        tee.close()
        sys.stdout = tee.stdout  # Restore original stdout
