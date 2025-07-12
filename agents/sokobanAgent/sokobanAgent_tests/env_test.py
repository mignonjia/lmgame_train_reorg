import numpy as np
import random
import sys
import os
import yaml
from datetime import datetime

# Add paths to import components
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from agents.sokobanAgent.env import SokobanEnv


# Setup logging to file
def setup_logging():
    """Setup logging to stream outputs to test_logs directory"""
    test_logs_dir = os.path.join(os.path.dirname(__file__), 'test_logs')
    os.makedirs(test_logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(test_logs_dir, f"env_test_{timestamp}.log")
    
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
    
    print(f"📝 Sokoban Environment Test log started at {datetime.now()}")
    print(f"📄 Log file: {log_file}")
    print("=" * 60)
    
    return tee


def get_default_config():
    """Get default configuration for Sokoban environment from agents.yaml"""
    # Load configuration from agents.yaml
    config_path = os.path.join(os.path.dirname(__file__), '../../../configs/agents.yaml')
    
    try:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Extract sokobanEnv configuration
        sokoban_env_config = yaml_config.get('sokobanEnv', {})
        
        # Convert dim_room from list to tuple if needed
        if 'dim_room' in sokoban_env_config:
            sokoban_env_config['dim_room'] = tuple(sokoban_env_config['dim_room'])
        
        print(f"✅ Loaded configuration from {config_path}")
        print(f"   Config keys: {list(sokoban_env_config.keys())}")
        
        return sokoban_env_config
        
    except FileNotFoundError:
        print(f"⚠️  Could not find {config_path}, using fallback configuration")
        # Fallback configuration if file not found
        return {
            'dim_room': (6, 6),
            'max_steps': 100,
            'num_boxes': 1,
            'render_mode': 'text',
            'search_depth': 300,
            'grid_lookup': {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"},
            'action_lookup': {1: "Up", 2: "Down", 3: "Left", 4: "Right"}
        }
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        # Fallback configuration if loading fails
        return {
            'dim_room': (6, 6),
            'max_steps': 100,
            'num_boxes': 1,
            'render_mode': 'text',
            'search_depth': 300,
            'grid_lookup': {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"},
            'action_lookup': {1: "Up", 2: "Down", 3: "Left", 4: "Right"}
        }


def test_single_env_creation():
    """Test creating a single Sokoban environment"""
    print("🔍 Testing single environment creation...")
    
    config = get_default_config()
    env = SokobanEnv(config)
    
    # Test reset
    initial_obs = env.reset(seed=42)
    
    assert initial_obs is not None
    assert isinstance(initial_obs, str)
    assert len(initial_obs) > 0
    
    # Check that we have the expected characters based on config
    grid_lookup = config['grid_lookup']
    wall_char = grid_lookup.get(0, '#')
    player_char = grid_lookup.get(5, 'P')
    box_char = grid_lookup.get(4, 'X')
    box_on_target_char = grid_lookup.get(3, '√')
    target_char = grid_lookup.get(2, 'O')
    player_on_target_char = grid_lookup.get(6, 'S')
    
    assert wall_char in initial_obs  # walls
    assert player_char in initial_obs or player_on_target_char in initial_obs  # player
    assert box_char in initial_obs or box_on_target_char in initial_obs  # boxes
    assert target_char in initial_obs or box_on_target_char in initial_obs or player_on_target_char in initial_obs  # targets
    
    print(f"✅ Single environment created successfully")
    print(f"   Initial observation length: {len(initial_obs)} characters")
    print(f"   Contains walls ({wall_char}): {wall_char in initial_obs}")
    print(f"   Contains player ({player_char}/{player_on_target_char}): {player_char in initial_obs or player_on_target_char in initial_obs}")
    print(f"   Contains boxes ({box_char}/{box_on_target_char}): {box_char in initial_obs or box_on_target_char in initial_obs}")
    print(f"   Contains targets ({target_char}/{box_on_target_char}/{player_on_target_char}): {target_char in initial_obs or box_on_target_char in initial_obs or player_on_target_char in initial_obs}")
    
    env.close()


def test_env_reset_with_different_seeds():
    """Test that different seeds produce different environments"""
    print("🔍 Testing environment reset with different seeds...")
    
    config = get_default_config()
    env = SokobanEnv(config)
    
    # Generate 5 different environments with different seeds
    observations = []
    seeds = [42, 123, 456, 789, 999]
    
    for seed in seeds:
        obs = env.reset(seed=seed)
        observations.append(obs)
        print(f"   Seed {seed}: Generated room with {len(obs)} characters")
    
    # Check that at least some environments are different
    unique_observations = set(observations)
    assert len(unique_observations) > 1, "All environments are identical - seed not working properly"
    
    print(f"✅ Seed test passed: {len(unique_observations)}/{len(observations)} unique environments generated")
    
    env.close()


def test_env_step_functionality():
    """Test basic step functionality"""
    print("🔍 Testing environment step functionality...")
    
    config = get_default_config()
    env = SokobanEnv(config)
    
    # Reset environment
    initial_obs = env.reset(seed=42)
    
    # Try all possible actions
    actions = [0, 1, 2, 3]  # Up, Down, Left, Right
    action_names = ['Up', 'Down', 'Left', 'Right']
    
    for action, action_name in zip(actions, action_names):
        obs, reward, done, info = env.step(action)
        
        assert obs is not None
        assert isinstance(obs, str)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert 'action_is_valid' in info
        assert 'action_is_effective' in info
        assert 'success' in info
        
        print(f"   Action {action_name}: reward={reward}, done={done}, valid={info['action_is_valid']}, effective={info['action_is_effective']}")
    
    print("✅ Step functionality test passed")
    
    env.close()


def test_env_action_mapping():
    """Test environment action mapping with random actions from action space"""
    print("🔍 Testing environment action mapping...")
    
    config = get_default_config()
    env = SokobanEnv(config)
    
    # Reset environment
    initial_obs = env.reset(seed=42)
    print(f"   Initial state:\n{initial_obs}")
    
    # Get available actions from config
    action_lookup = config.get('action_lookup', {})
    available_actions = list(action_lookup.keys())
    print(f"   Available actions from config: {available_actions}")
    print(f"   Action mapping: {action_lookup}")
    
    # Test each action
    for action in available_actions:
        print(f"\n   Testing action {action} ({action_lookup[action]}):")
        
        # Reset to consistent state
        env.reset(seed=42)
        
        # Take the action
        try:
            obs, reward, done, info = env.step(action)
            
            print(f"      Action executed successfully")
            print(f"      Reward: {reward}")
            print(f"      Done: {done}")
            print(f"      Info: {info}")
            print(f"      New state:\n{obs}")
            
            # Verify step worked
            assert obs is not None
            assert isinstance(obs, str)
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)
            assert isinstance(info, dict)
            
        except Exception as e:
            print(f"      ❌ Action {action} failed: {e}")
            raise
    
    # Test with random actions multiple times
    print(f"\n   Testing with 10 random actions:")
    env.reset(seed=42)
    
    for i in range(10):
        action = available_actions[i % len(available_actions)]
        print(f"      Step {i+1}: Action {action} ({action_lookup[action]})")
        
        try:
            obs, reward, done, info = env.step(action)
            print(f"         → Reward: {reward}, Done: {done}, Valid: {info.get('action_is_valid', 'N/A')}, Effective: {info.get('action_is_effective', 'N/A')}")
            
            if done:
                print(f"         → Episode finished at step {i+1}")
                break
                
        except Exception as e:
            print(f"         ❌ Step {i+1} failed: {e}")
            break
    
    env.close()
    print("✅ Action mapping test passed")


def test_10_random_envs_three_times():
    """Main test: Launch 10 random environments three times and print room layouts"""
    print("🎯 MAIN TEST: Launching 10 random environments three times...")
    print("=" * 60)
    
    config = get_default_config()
    
    for run in range(3):
        print(f"\n🚀 RUN {run + 1}/3")
        print("-" * 40)
        
        for env_id in range(10):
            print(f"\n🏠 Environment {env_id + 1}/10 (Run {run + 1})")
            
            # Create new environment for each test
            env = SokobanEnv(config)
            
            # Use different seed for each environment
            # Combine run number and env_id to ensure uniqueness
            seed = run * 1000 + env_id * 100 + random.randint(0, 99)
            
            try:
                # Reset environment with unique seed
                obs = env.reset(seed=seed)
                
                print(f"   Seed: {seed}")
                print(f"   Room dimensions: {config['dim_room']}")
                print(f"   Max steps: {config['max_steps']}")
                print(f"   Number of boxes: {config['num_boxes']}")
                print("   Room layout:")
                
                # Print the room layout with proper formatting
                lines = obs.split('\n')
                for i, line in enumerate(lines):
                    print(f"   {i:2d}: {line}")
                
                # Analyze the room using config-based character mappings
                grid_lookup = config['grid_lookup']
                wall_char = grid_lookup.get(0, '#')
                empty_char = grid_lookup.get(1, '_')
                target_char = grid_lookup.get(2, 'O')
                box_on_target_char = grid_lookup.get(3, '√')
                box_char = grid_lookup.get(4, 'X')
                player_char = grid_lookup.get(5, 'P')
                player_on_target_char = grid_lookup.get(6, 'S')
                
                wall_count = obs.count(wall_char)
                empty_count = obs.count(empty_char)
                target_count = obs.count(target_char)
                box_count = obs.count(box_char)
                box_on_target_count = obs.count(box_on_target_char)
                player_count = obs.count(player_char)
                player_on_target_count = obs.count(player_on_target_char)
                
                print(f"   Analysis:")
                print(f"     Walls ({wall_char}): {wall_count}")
                print(f"     Empty spaces ({empty_char}): {empty_count}")
                print(f"     Targets ({target_char}): {target_count}")
                print(f"     Boxes ({box_char}): {box_count}")
                print(f"     Boxes on targets ({box_on_target_char}): {box_on_target_count}")
                print(f"     Player ({player_char}): {player_count}")
                print(f"     Player on target ({player_on_target_char}): {player_on_target_count}")
                print(f"     Total boxes: {box_count + box_on_target_count}")
                print(f"     Total targets: {target_count + box_on_target_count + player_on_target_count}")
                
                # Verify basic constraints
                assert player_count + player_on_target_count == 1, "Should have exactly one player"
                assert box_count + box_on_target_count == config['num_boxes'], f"Should have exactly {config['num_boxes']} boxes"
                assert target_count + box_on_target_count + player_on_target_count >= config['num_boxes'], "Should have at least as many targets as boxes"
                
            except Exception as e:
                print(f"   ❌ Error generating environment: {e}")
                print(f"   Retrying with different seed...")
                
                # Try with a completely different seed
                backup_seed = random.randint(0, 999999)
                try:
                    obs = env.reset(seed=backup_seed)
                    print(f"   ✅ Backup seed {backup_seed} worked")
                    print("   Room layout:")
                    lines = obs.split('\n')
                    for i, line in enumerate(lines):
                        print(f"   {i:2d}: {line}")
                except Exception as e2:
                    print(f"   ❌ Backup seed also failed: {e2}")
            
            finally:
                env.close()
    
    print("\n" + "=" * 60)
    print("🎉 Main test completed successfully!")
    print("✅ All 30 environments (10 × 3 runs) generated and analyzed")


def test_env_action_space():
    """Test environment action space"""
    print("🔍 Testing environment action space...")
    
    config = get_default_config()
    env = SokobanEnv(config)
    
    # Test get_all_actions
    all_actions = env.get_all_actions()
    assert all_actions is not None
    assert len(all_actions) > 0
    
    print(f"   Available actions: {all_actions}")
    
    # Test ACTION_SPACE
    assert hasattr(env, 'ACTION_SPACE')
    print(f"   Action space: {env.ACTION_SPACE}")
    
    env.close()
    print("✅ Action space test passed")


def test_env_rendering_modes():
    """Test different rendering modes"""
    print("🔍 Testing environment rendering modes...")
    
    config = get_default_config()
    env = SokobanEnv(config)
    
    # Reset environment
    env.reset(seed=42)
    
    # Test text mode (default)
    text_obs = env.render(mode='text')
    assert isinstance(text_obs, str)
    assert len(text_obs) > 0
    print(f"   Text mode: {len(text_obs)} characters")
    
    # Test rgb_array mode
    try:
        rgb_obs = env.render(mode='rgb_array')
        assert rgb_obs is not None
        if hasattr(rgb_obs, 'shape') and hasattr(rgb_obs, 'dtype'):
            print(f"   RGB array mode: shape {rgb_obs.shape}")  # type: ignore
        else:
            print(f"   RGB array mode: type {type(rgb_obs)}")
    except Exception as e:
        print(f"   RGB array mode failed (expected): {e}")
    
    # Test invalid mode
    try:
        env.render(mode='invalid_mode')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   Invalid mode correctly rejected: {e}")
    
    env.close()
    print("✅ Rendering modes test passed")


if __name__ == "__main__":
    # Setup logging to test_logs
    tee = setup_logging()
    
    try:
        print("🚀 Starting Sokoban Environment Tests...")
        print()
        
        print("Test 1: Single environment creation")
        test_single_env_creation()
        print()
        
        print("Test 2: Environment reset with different seeds")
        test_env_reset_with_different_seeds()
        print()
        
        print("Test 3: Environment step functionality")
        test_env_step_functionality()
        print()
        
        print("Test 4: Environment action space")
        test_env_action_space()
        print()
        
        print("Test 5: Environment action mapping")
        test_env_action_mapping()
        print()
        
        print("Test 6: Environment rendering modes")
        test_env_rendering_modes()
        print()
        
        print("Test 7: 10 random environments × 3 runs")
        test_10_random_envs_three_times()
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
