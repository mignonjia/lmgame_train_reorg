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
    """Simple mock agent that generates prompts using Qwen tokenizer"""
    
    def __init__(self, agent_id, tokenizer):
        self.agent_id = agent_id
        self.tokenizer = tokenizer
    
    def get_llm_prompts(self, env_out):
        """Return a single-prompt DataProto with real tokenized text following VERL pattern"""
        prompt_text = f"Agent {self.agent_id}: What is 2+2?"
        
        # Import VERL utilities to match their pattern
        import verl.utils.torch_functional as verl_F
        from verl.utils.model import compute_position_id_with_mask
        
        # Use VERL's tokenization function (similar to QwenVLRolloutManager)
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_text,
            tokenizer=self.tokenizer,
            max_length=128,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,  # Following VERL pattern
            truncation="right"  # VERL expects: "left", "right", "middle", or "error"
        )
        
        # Compute position_ids using VERL's utility (like in QwenVLRolloutManager)
        position_ids = compute_position_id_with_mask(attention_mask)
        
        # Remove batch dimension for single prompt (input_ids comes as [1, seq_len])
        input_ids = input_ids.squeeze(0)  # [seq_len]
        attention_mask = attention_mask.squeeze(0)  # [seq_len]
        position_ids = position_ids.squeeze(0)  # [seq_len]
        
        # Re-add batch dimension and create TensorDict for VERL DataProto
        batch_tensordict = TensorDict({
            "input_ids": input_ids.unsqueeze(0),  # [1, seq_len]
            "attention_mask": attention_mask.unsqueeze(0),  # [1, seq_len]
            "position_ids": position_ids.unsqueeze(0)  # [1, seq_len]
        }, batch_size=[1])  # Single prompt batch
        
        return DataProto(
            batch=batch_tensordict,
            non_tensor_batch={
                "raw_prompt": np.array([prompt_text], dtype=object)
            },
            meta_info={"temperature": 1.0}
        )
    
    def get_initial_env_outputs(self):
        return Mock(done=False)


class MockConfig:
    """Simple mock config"""
    def __init__(self):
        self.agent_batch_size = 2
        self.n_gpus_per_node = 1
        self.train = ["sokobanAgent"]


def test_collect_prompts_basic():
    """Test _collect_prompts combines single DataProtos into batch correctly"""
    
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
    
    # Test _collect_prompts
    batched_dataproto, idx_map = rollout._collect_prompts()
    
    # Verify results
    assert batched_dataproto is not None
    assert idx_map == [0, 1]
    
    # Check batch dimensions - all three required elements
    assert batched_dataproto.batch["input_ids"].shape[0] == 2  # 2 agents
    assert batched_dataproto.batch["attention_mask"].shape[0] == 2
    assert batched_dataproto.batch["position_ids"].shape[0] == 2
    
    # Check raw prompts
    raw_prompts = batched_dataproto.non_tensor_batch["raw_prompt"]
    assert len(raw_prompts) == 2
    assert "Agent 0" in raw_prompts[0]
    assert "Agent 1" in raw_prompts[1]
    
    print("✅ Basic collect_prompts test passed")
    
    # Additional verification: Test roundtrip consistency
    # single prompt → single DataProto → batch DataProto → single DataProto → single prompt
    
    # Step 1: Get individual DataProtos
    agent_0_dataproto = rollout.agents[0].get_llm_prompts(rollout.env_outs[0])
    agent_1_dataproto = rollout.agents[1].get_llm_prompts(rollout.env_outs[1])
    
    # Step 2: Verify batch contains both prompts correctly
    batch_prompts = batched_dataproto.non_tensor_batch["raw_prompt"]
    original_prompts = [
        agent_0_dataproto.non_tensor_batch["raw_prompt"][0],
        agent_1_dataproto.non_tensor_batch["raw_prompt"][0]
    ]
    
    assert batch_prompts[0] == original_prompts[0]
    assert batch_prompts[1] == original_prompts[1]
    
    # Step 3: Extract individual prompts from batch (batch → single)
    batch_input_ids = batched_dataproto.batch["input_ids"]
    batch_attention_mask = batched_dataproto.batch["attention_mask"]
    batch_position_ids = batched_dataproto.batch["position_ids"]
    
    # Step 4: Decode back to text and verify alignment
    reconstructed_prompt_0 = tokenizer.decode(
        batch_input_ids[0], 
        skip_special_tokens=True
    ).strip()
    
    reconstructed_prompt_1 = tokenizer.decode(
        batch_input_ids[1], 
        skip_special_tokens=True
    ).strip()
    
    # Verify the core content is preserved (tokenization may add/remove some formatting)
    assert "Agent 0" in reconstructed_prompt_0
    assert "Agent 1" in reconstructed_prompt_1
    assert "What is 2+2?" in reconstructed_prompt_0
    assert "What is 2+2?" in reconstructed_prompt_1
    
    print("✅ Roundtrip consistency verified: single prompt → DataProto → batch → single prompt")


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
    
    # Test _collect_prompts
    batched_dataproto, idx_map = rollout._collect_prompts()
    
    # Should only have second agent
    assert idx_map == [1]
    assert batched_dataproto is not None
    assert batched_dataproto.batch["input_ids"].shape[0] == 1
    assert "Agent 1" in batched_dataproto.non_tensor_batch["raw_prompt"][0]
    
    print("✅ Done agent test passed")


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
    
    # Test _collect_prompts
    result = rollout._collect_prompts()
    
    # Should return None, None
    assert result == (None, None)
    
    print("✅ All done test passed")


if __name__ == "__main__":
    # Setup logging to test_logs
    tee = setup_logging()
    
    try:
        print("🚀 Starting _collect_prompts tests...")
        print()
        
        print("Test 1: Basic functionality")
        test_collect_prompts_basic()
        print()
        
        print("Test 2: Done agent handling")
        test_collect_prompts_with_done_agent()
        print()
        
        print("Test 3: All done scenario")
        test_collect_prompts_all_done()
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
