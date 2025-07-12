#!/usr/bin/env python3
"""
SokobanAgent Test - Tests agent rollout logic with mocked LLM responses
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

from agents.sokobanAgent.agent import SokobanAgent


# Setup logging to file
def setup_logging():
    """Setup logging to stream outputs to test_logs directory"""
    test_logs_dir = os.path.join(os.path.dirname(__file__), 'test_logs')
    os.makedirs(test_logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(test_logs_dir, f"agent_test_{timestamp}.log")
    
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
    
    print(f"📝 SokobanAgent Test log started at {datetime.now()}")
    print(f"📄 Log file: {log_file}")
    print("=" * 60)
    
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

def get_mock_llm_responses():
    """Generate mock LLM responses for testing with valid actions"""
    mock_responses = [
        "<answer>Right || Down</answer>",
        "<answer>Left || Up || Right</answer>", 
        "<answer>Down</answer>",
        "<answer>Up || Left</answer>",
        "<answer>Right || Right || Down</answer>",
        "<answer>Up</answer>",
        "<answer>Down || Left</answer>",
        "<answer>Right</answer>"
    ]
    return mock_responses

def test_sokoban_agent_creation():
    """Test SokobanAgent creation and basic functionality"""
    print("🔍 Testing SokobanAgent creation...")
    
    # Load configuration
    config = load_config()
    
    # Create agent
    agent = SokobanAgent(
        config=config,
        group_id=0,
        agent_id=0,
        seed=42,
        tag="TestSokoban"
    )
    
    # Verify basic properties
    assert agent.agent_id == 0
    assert agent.group_id == 0
    assert agent.tag == "TestSokoban"
    assert agent.max_turns > 0
    assert agent.max_actions_all_turns > 0
    assert hasattr(agent, 'env')
    assert hasattr(agent, 'messages')
    assert hasattr(agent, 'trajectory_history')
    
    print(f"✅ SokobanAgent created successfully")
    print(f"   Agent ID: {agent.agent_id}")
    print(f"   Group ID: {agent.group_id}")
    print(f"   Tag: {agent.tag}")
    print(f"   Max turns: {agent.max_turns}")
    print(f"   Max actions all turns: {agent.max_actions_all_turns}")
    print(f"   Seed: {agent.seed}")
    
    agent.close()


def test_sokoban_agent_reset():
    """Test SokobanAgent reset functionality"""
    print("🔍 Testing SokobanAgent reset...")
    
    config = load_config()
    agent = SokobanAgent(config=config, agent_id=0, group_id=0, seed=42, tag="TestSokoban")
    
    # Test reset
    initial_env_outputs = agent.reset(seed=42)
    
    # Verify reset returns proper EnvOutput
    assert initial_env_outputs is not None
    assert hasattr(initial_env_outputs, 'done')
    assert hasattr(initial_env_outputs, 'state')
    assert hasattr(initial_env_outputs, 'reward')
    assert hasattr(initial_env_outputs, 'info')
    
    assert initial_env_outputs.done == False
    assert isinstance(initial_env_outputs.state, str)
    assert len(initial_env_outputs.state) > 0
    assert initial_env_outputs.reward == 0.0
    assert isinstance(initial_env_outputs.info, dict)
    
    print(f"✅ Reset test passed")
    print(f"   Done: {initial_env_outputs.done}")
    print(f"   Reward: {initial_env_outputs.reward}")
    print(f"   State length: {len(initial_env_outputs.state)} characters")
    print(f"   Info keys: {list(initial_env_outputs.info.keys())}")
    
    agent.close()


def test_sokoban_agent_action_extraction():
    """Test action extraction from LLM responses"""
    print("🔍 Testing action extraction from LLM responses...")
    
    config = load_config()
    agent = SokobanAgent(config=config, agent_id=0, group_id=0, seed=42, tag="TestSokoban")
    
    # Test various LLM response formats
    test_cases = [
        ("<answer>Right || Down</answer>", ["Right", "Down"]),
        ("<answer>Left || Up || Right</answer>", ["Left", "Up", "Right"]),
        ("<answer>Down</answer>", ["Down"]),
        ("<answer>Up || Left</answer>", ["Up", "Left"]),
        ("<answer>Right || Right || Down</answer>", ["Right", "Right", "Down"]),
        ("I think the best move is <answer>Up</answer>", ["Up"]),
        ("Let me analyze... <answer>Down || Left</answer> should work", ["Down", "Left"]),
    ]
    
    # Reset agent
    env_outputs = agent.reset(seed=42)
    
    print(f"   Testing {len(test_cases)} different LLM response formats:")
    
    for i, (llm_response, expected_actions) in enumerate(test_cases):
        print(f"\n   Test case {i+1}:")
        print(f"      LLM Response: {llm_response}")
        print(f"      Expected Actions: {expected_actions}")
        
        # Process the response
        env_outputs = agent.get_env_outputs(llm_response)
        
        # Get the last trajectory entry to see what actions were executed
        if agent.trajectory_history:
            last_traj = agent.trajectory_history[-1]
            executed_actions = last_traj.actions
            print(f"      Executed Actions: {executed_actions}")
            print(f"      Actions Left: {last_traj.actions_left}")
            print(f"      Total Actions Consumed: {agent.total_actions_consumed}")
            print(f"      Reward: {last_traj.reward}")
            print(f"      Info: {last_traj.info}")
        
        # Check if we've reached limits
        if env_outputs.done:
            print(f"      Episode done: {env_outputs.done}")
            break
    
    print(f"\n   ✅ Action extraction test completed")
    print(f"      Total trajectory steps: {len(agent.trajectory_history)}")
    print(f"      Final total actions consumed: {agent.total_actions_consumed}")
    print(f"      Max actions allowed: {agent.max_actions_all_turns}")
    
    agent.close()


def test_sokoban_agent_rollout():
    """Test SokobanAgent with complete rollout logic"""
    print("🔍 Testing SokobanAgent rollout logic...")
    
    # Load configuration
    config = load_config()
    
    # Create agent
    agent = SokobanAgent(
        config=config,
        group_id=0,
        agent_id=0,
        seed=42,
        tag="TestSokoban"
    )
    
    # Get mock LLM responses
    mock_responses = get_mock_llm_responses()
    print(f"   Generated {len(mock_responses)} mock LLM responses")
    
    print("\n   🚀 STARTING ROLLOUT")
    print("   " + "=" * 50)
    
    # Initial reset - get initial env outputs
    initial_env_outputs = agent.reset(seed=42)
    print(f"\n   🔄 RESET - Initial Environment State:")
    print(f"      Done: {initial_env_outputs.done}")
    print(f"      Reward: {initial_env_outputs.reward}")
    print(f"      State:\n{initial_env_outputs.state}")
    
    # Rollout loop
    env_outputs = initial_env_outputs
    response_idx = 0
    
    for cur_turn in range(agent.max_turns):
        print(f"\n   {'='*15} TURN {cur_turn + 1} {'='*15}")
        
        # Check if done
        if env_outputs.done:
            print(f"   🏁 Episode finished early at turn {cur_turn + 1}")
            break
        
        # Get LLM prompts
        llm_prompts = agent.get_llm_prompts(env_outputs)
        print(f"   📝 LLM Prompts ({len(llm_prompts)} messages):")
        for i, msg in enumerate(llm_prompts):
            role = msg['role']
            content = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
            print(f"      {i+1}. {role}: {content}")
        
        # Mock LLM response
        if response_idx < len(mock_responses):
            llm_response = mock_responses[response_idx]
            response_idx += 1
        else:
            llm_response = "<answer>Right</answer>"  # Default fallback
        
        print(f"   🤖 Mock LLM Response: {llm_response}")
        
        # Get environment outputs
        env_outputs = agent.get_env_outputs(llm_response)
        print(f"   🌍 Environment Response:")
        print(f"      Done: {env_outputs.done}")
        print(f"      Reward: {env_outputs.reward}")
        print(f"      Info: {env_outputs.info}")
        print(f"      State:\n{env_outputs.state}")
        
        # Check if episode is done
        if env_outputs.done:
            print(f"   🏁 Episode completed at turn {cur_turn + 1}")
            break
    
    print(f"\n   ✅ Rollout completed successfully")
    print(f"      Total turns: {len(agent.trajectory_history)}")
    print(f"      Final done status: {env_outputs.done}")
    

def test_sokoban_agent_final_rollout_states():
    """Test final rollout states collection"""
    print("🔍 Testing final rollout states collection...")
    
    config = load_config()
    agent = SokobanAgent(config=config, agent_id=0, group_id=0, seed=42, tag="TestSokoban")
    
    # Run a short rollout
    env_outputs = agent.reset(seed=42)
    mock_responses = get_mock_llm_responses()
    
    # Execute a few turns
    for turn in range(min(3, agent.max_turns)):
        if env_outputs.done:
            break
        
        agent.get_llm_prompts(env_outputs)
        llm_response = mock_responses[turn] if turn < len(mock_responses) else "<answer>Right</answer>"
        env_outputs = agent.get_env_outputs(llm_response)
    
    # Get final rollout states
    final_rollout_states = agent.get_final_rollout_states()
    
    # Verify structure
    assert isinstance(final_rollout_states, dict)
    assert 'env_id' in final_rollout_states
    assert 'history' in final_rollout_states
    assert 'group_id' in final_rollout_states
    assert 'tag' in final_rollout_states
    assert 'metrics' in final_rollout_states
    
    assert final_rollout_states['env_id'] == 0
    assert final_rollout_states['group_id'] == 0
    assert final_rollout_states['tag'] == "TestSokoban"
    assert isinstance(final_rollout_states['history'], list)
    assert isinstance(final_rollout_states['metrics'], dict)
    
    print(f"✅ Final rollout states test passed")
    print(f"   Environment ID: {final_rollout_states['env_id']}")
    print(f"   Group ID: {final_rollout_states['group_id']}")
    print(f"   Tag: {final_rollout_states['tag']}")
    print(f"   History Length: {len(final_rollout_states['history'])}")
    print(f"   Metrics: {list(final_rollout_states['metrics'].keys())}")
    
    agent.close()


def test_sokoban_agent_complete_rollout():
    """Test complete rollout with detailed output"""
    print("🔍 Testing complete rollout with detailed output...")
    
    config = load_config()
    agent = SokobanAgent(config=config, agent_id=0, group_id=0, seed=42, tag="TestSokoban")
    
    mock_responses = get_mock_llm_responses()
    
    # Initial reset
    initial_env_outputs = agent.reset(seed=42)
    print(f"   🔄 Initial state: Done={initial_env_outputs.done}, Reward={initial_env_outputs.reward}")
    
    # Rollout loop
    env_outputs = initial_env_outputs
    response_idx = 0
    
    for cur_turn in range(agent.max_turns):
        if env_outputs.done:
            print(f"   🏁 Episode finished early at turn {cur_turn + 1}")
            break
        
        # Get LLM prompts
        llm_prompts = agent.get_llm_prompts(env_outputs)
        
        # Mock LLM response
        if response_idx < len(mock_responses):
            llm_response = mock_responses[response_idx]
            response_idx += 1
        else:
            llm_response = "<answer>Right</answer>"
        
        print(f"   Turn {cur_turn + 1}: {llm_response}")
        
        # Get environment outputs
        env_outputs = agent.get_env_outputs(llm_response)
        print(f"   → Done: {env_outputs.done}, Reward: {env_outputs.reward}")
        
        if env_outputs.done:
            print(f"   🏁 Episode completed at turn {cur_turn + 1}")
            break
    
    # Get final rollout states
    final_rollout_states = agent.get_final_rollout_states()
    
    print(f"\n   📊 FINAL ROLLOUT STATES:")
    print(f"      Environment ID: {final_rollout_states['env_id']}")
    print(f"      Group ID: {final_rollout_states['group_id']}")
    print(f"      Tag: {final_rollout_states['tag']}")
    print(f"      History Length: {len(final_rollout_states['history'])}")
    
    print(f"\n   📈 METRICS:")
    for key, value in final_rollout_states['metrics'].items():
        print(f"      {key}: {value}")
    
    print(f"\n   📜 TRAJECTORY HISTORY:")
    for i, step in enumerate(final_rollout_states['history']):
        print(f"\n      Step {i+1}:")
        print(f"         Actions Left: {step['actions_left']}")
        print(f"         Actions: {step['actions']}")
        print(f"         Reward: {step['reward']}")
        print(f"         Info: {step['info']}")
        print(f"         LLM Response: {step['llm_response']}")
        print(f"         LLM Raw Response: {step['llm_raw_response']}")
        state_lines = step['state'].split('\n')
        print(f"         State:")
        for line in state_lines:
            print(f"           {line}")
    
    print(f"\n   💬 COMPLETE MESSAGE HISTORY:")
    for i, msg in enumerate(agent.messages):
        role = msg['role']
        content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
        print(f"      Message {i+1} ({role}): {content}")
    
    print(f"\n   📄 FINAL ROLLOUT STATES (JSON):")
    print(json.dumps(final_rollout_states, indent=4, default=str))
    
    agent.close()
    print(f"   ✅ Complete rollout test passed")

if __name__ == "__main__":
    # Setup logging to test_logs
    tee = setup_logging()
    
    try:
        print("🚀 Starting SokobanAgent Tests...")
        print()
        
        print("Test 1: Agent creation")
        test_sokoban_agent_creation()
        print()
        
        print("Test 2: Agent reset functionality")
        test_sokoban_agent_reset()
        print()
        
        print("Test 3: Action extraction from LLM responses")
        test_sokoban_agent_action_extraction()
        print()
        
        print("Test 4: Agent rollout logic")
        test_sokoban_agent_rollout()
        print()
        
        print("Test 5: Final rollout states")
        test_sokoban_agent_final_rollout_states()
        print()
        
        print("Test 6: Complete rollout with detailed output")
        test_sokoban_agent_complete_rollout()
        print()
        
        print("=" * 60)
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
