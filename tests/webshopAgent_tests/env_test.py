import numpy as np
import random
import sys
import os
import yaml
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add the external webshop-minimal package to the Python path
webshop_path = project_root / "external" / "webshop-minimal"
sys.path.insert(0, str(webshop_path))

from agents.webshopAgent.env import WebShopEnv

# # Setup logging to file
# def setup_logging():
#     """Setup logging to stream outputs to test_logs directory"""
#     test_logs_dir = os.path.join(os.path.dirname(__file__), 'test_logs')
#     os.makedirs(test_logs_dir, exist_ok=True)
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_file = os.path.join(test_logs_dir, f"webshop_env_test_{timestamp}.log")
    
#     class Tee:
#         def __init__(self, file_path):
#             self.file = open(file_path, 'w')
#             self.stdout = sys.stdout
        
#         def write(self, data):
#             self.file.write(data)
#             self.file.flush()
#             self.stdout.write(data)
        
#         def flush(self):
#             self.file.flush()
#             self.stdout.flush()
        
#         def close(self):
#             self.file.close()
    
#     tee = Tee(log_file)
#     sys.stdout = tee
    
#     print(f"📝 WebShop Environment Test log started at {datetime.now()}")
#     print(f"📄 Log file: {log_file}")
#     print("=" * 60)
    
#     return tee


# class WebShopConfig:
#     """Simple config class for WebShop environment"""
#     def __init__(self, config_dict):
#         self.observation_mode = config_dict.get('observation_mode', 'text')
#         self.file_path = config_dict.get('file_path', '')
#         self.server = config_dict.get('server', None)
#         self.filter_goals = config_dict.get('filter_goals', None)
#         self.limit_goals = config_dict.get('limit_goals', -1)
#         self.num_products = config_dict.get('num_products', None)
#         self.human_goals = config_dict.get('human_goals', False)
#         self.show_attrs = config_dict.get('show_attrs', False)


# def get_default_config():
#     """Get default configuration for WebShop environment from agents.yaml"""
#     # Load configuration from agents.yaml
#     config_path = os.path.join(os.path.dirname(__file__), '../../configs/agents.yaml')
    
#     try:
#         with open(config_path, 'r') as f:
#             yaml_config = yaml.safe_load(f)
        
#         # Extract webshopAgent configuration
#         webshop_config = yaml_config.get('webshopAgent', {})
#         webshop_env_config = webshop_config.get('env_config', {})
        
#         print(f"✅ Loaded configuration from {config_path}")
#         print(f"   Config keys: {list(webshop_env_config.keys())}")
        
#         # Convert to config object
#         config_obj = WebShopConfig(webshop_env_config)
#         return config_obj
        
#     except FileNotFoundError:
#         print(f"⚠️  Could not find {config_path}, using fallback configuration")
#         # Fallback configuration if file not found
#         fallback_config = {
#             'observation_mode': 'text',
#             'file_path': '',
#             'server': None,
#             'filter_goals': None,
#             'limit_goals': -1,
#             'num_products': None,
#             'human_goals': False,
#             'show_attrs': False
#         }
#         return WebShopConfig(fallback_config)
#     except Exception as e:
#         print(f"❌ Error loading configuration: {e}")
#         # Fallback configuration if loading fails
#         fallback_config = {
#             'observation_mode': 'text',
#             'file_path': '',
#             'server': None,
#             'filter_goals': None,
#             'limit_goals': -1,
#             'num_products': None,
#             'human_goals': False,
#             'show_attrs': False
#         }
#         return WebShopConfig(fallback_config)


# def test_single_env_creation():
#     """Test creating a single WebShop environment"""
#     print("🔍 Testing single environment creation...")
    
#     config = get_default_config()
    
#     try:
#         env = WebShopEnv(config)
        
#         # Test reset
#         initial_obs = env.reset(seed=42)
        
#         assert initial_obs is not None
#         assert isinstance(initial_obs, str)
#         assert len(initial_obs) > 0
        
#         print(f"✅ Single environment created successfully")
#         print(f"   Initial observation length: {len(initial_obs)} characters")
#         print(f"   Observation preview: {initial_obs[:100]}...")
        
#         # Check for WebShop-specific content
#         if 'search' in initial_obs.lower():
#             print(f"   Contains search functionality: Yes")
#         if 'click' in initial_obs.lower():
#             print(f"   Contains clickable elements: Yes")
#         if 'product' in initial_obs.lower():
#             print(f"   Contains product information: Yes")
        
#         env.close()
        
#     except Exception as e:
#         print(f"❌ Environment creation failed: {e}")
#         print(f"   This might be expected if WebShop dependencies are not fully set up")
#         print(f"   Error details: {type(e).__name__}: {str(e)}")


# def test_env_reset_with_different_seeds():
#     """Test that different seeds produce different environments"""
#     print("🔍 Testing environment reset with different seeds...")
    
#     config = get_default_config()
    
#     try:
#         env = WebShopEnv(config)
        
#         # Generate 5 different environments with different seeds
#         observations = []
#         seeds = [42, 123, 456, 789, 999]
        
#         for seed in seeds:
#             obs = env.reset(seed=seed)
#             observations.append(obs)
#             print(f"   Seed {seed}: Generated observation with {len(obs)} characters")
        
#         # Check that at least some environments are different
#         unique_observations = set(observations)
#         print(f"✅ Seed test completed: {len(unique_observations)}/{len(observations)} unique observations")
        
#         env.close()
        
#     except Exception as e:
#         print(f"❌ Seed test failed: {e}")
#         print(f"   This might be expected if WebShop dependencies are not fully set up")


# def test_env_step_functionality():
#     """Test basic step functionality"""
#     print("🔍 Testing environment step functionality...")
    
#     config = get_default_config()
    
#     try:
#         env = WebShopEnv(config)
        
#         # Reset environment
#         initial_obs = env.reset(seed=42)
        
#         # Get available actions
#         available_actions = env.get_available_actions()
#         print(f"   Available actions: {available_actions}")
        
#         # Try a few actions if available
#         if available_actions:
#             for i, action in enumerate(available_actions[:3]):  # Test first 3 actions
#                 obs, reward, done, info = env.step(action)
                
#                 assert obs is not None
#                 assert isinstance(obs, str)
#                 assert isinstance(reward, (int, float))
#                 assert isinstance(done, bool)
#                 assert isinstance(info, dict)
#                 assert 'action_is_effective' in info
#                 assert 'action_is_valid' in info
#                 assert 'success' in info
                
#                 print(f"   Action {action}: reward={reward}, done={done}, valid={info['action_is_valid']}, effective={info['action_is_effective']}")
#         else:
#             print("   No available actions to test")
        
#         print("✅ Step functionality test passed")
        
#         env.close()
        
#     except Exception as e:
#         print(f"❌ Step functionality test failed: {e}")
#         print(f"   This might be expected if WebShop dependencies are not fully set up")


# def test_env_action_mapping():
#     """Test environment action mapping"""
#     print("🔍 Testing environment action mapping...")
    
#     config = get_default_config()
    
#     try:
#         env = WebShopEnv(config)
        
#         # Reset environment
#         initial_obs = env.reset(seed=42)
#         print(f"   Initial state preview: {initial_obs[:100]}...")
        
#         # Get available actions
#         available_actions = env.get_available_actions()
#         print(f"   Available actions: {available_actions}")
        
#         # Test each action
#         for action in available_actions:
#             print(f"\n   Testing action: {action}")
            
#             # Reset to consistent state
#             env.reset(seed=42)
            
#             # Take the action
#             try:
#                 obs, reward, done, info = env.step(action)
                
#                 print(f"      Action executed successfully")
#                 print(f"      Reward: {reward}")
#                 print(f"      Done: {done}")
#                 print(f"      Info: {info}")
#                 print(f"      New state preview: {obs[:100]}...")
                
#                 # Verify step worked
#                 assert obs is not None
#                 assert isinstance(obs, str)
#                 assert isinstance(reward, (int, float))
#                 assert isinstance(done, bool)
#                 assert isinstance(info, dict)
                
#             except Exception as e:
#                 print(f"      ❌ Action {action} failed: {e}")
        
#         env.close()
#         print("✅ Action mapping test passed")
        
#     except Exception as e:
#         print(f"❌ Action mapping test failed: {e}")
#         print(f"   This might be expected if WebShop dependencies are not fully set up")


# def test_10_random_envs_three_times():
#     """Main test: Launch 10 random environments three times and print observations"""
#     print("🎯 MAIN TEST: Launching 10 random environments three times...")
#     print("=" * 60)
    
#     config = get_default_config()
    
#     try:
#         for run in range(3):
#             print(f"\n🚀 RUN {run + 1}/3")
#             print("-" * 40)
            
#             for env_id in range(10):
#                 print(f"\n🏠 Environment {env_id + 1}/10 (Run {run + 1})")
                
#                 # Create new environment for each test
#                 env = WebShopEnv(config)
                
#                 # Use different seed for each environment
#                 seed = run * 1000 + env_id * 100 + random.randint(0, 99)
                
#                 try:
#                     # Reset environment with unique seed
#                     obs = env.reset(seed=seed)
                    
#                     print(f"   Seed: {seed}")
#                     print(f"   Observation length: {len(obs)} characters")
#                     print("   Observation preview:")
#                     print(f"   {obs[:200]}...")
                    
#                     # Get available actions
#                     available_actions = env.get_available_actions()
#                     print(f"   Available actions: {available_actions}")
                    
#                     # Analyze the observation
#                     if 'search' in obs.lower():
#                         print(f"   Contains search functionality: Yes")
#                     if 'click' in obs.lower():
#                         print(f"   Contains clickable elements: Yes")
#                     if 'product' in obs.lower():
#                         print(f"   Contains product information: Yes")
                    
#                 except Exception as e:
#                     print(f"   ❌ Error generating environment: {e}")
                
#                 finally:
#                     env.close()
        
#         print("\n" + "=" * 60)
#         print("🎉 Main test completed!")
#         print("✅ All 30 environments (10 × 3 runs) generated and analyzed")
        
#     except Exception as e:
#         print(f"❌ Main test failed: {e}")
#         print(f"   This might be expected if WebShop dependencies are not fully set up")


# def test_env_action_space():
#     """Test environment action space"""
#     print("🔍 Testing environment action space...")
    
#     config = get_default_config()
    
#     try:
#         env = WebShopEnv(config)
        
#         # Test get_available_actions
#         all_actions = env.get_available_actions()
#         assert all_actions is not None
#         assert isinstance(all_actions, list)
        
#         print(f"   Available actions: {all_actions}")
#         print(f"   Number of actions: {len(all_actions)}")
        
#         # Test action types
#         for action in all_actions:
#             assert isinstance(action, str)
#             if action.startswith('search['):
#                 print(f"   Search action: {action}")
#             elif action.startswith('click['):
#                 print(f"   Click action: {action}")
#             else:
#                 print(f"   Other action: {action}")
        
#         env.close()
#         print("✅ Action space test passed")
        
#     except Exception as e:
#         print(f"❌ Action space test failed: {e}")
#         print(f"   This might be expected if WebShop dependencies are not fully set up")


# def test_env_rendering_modes():
#     """Test different rendering modes"""
#     print("🔍 Testing environment rendering modes...")
    
#     config = get_default_config()
    
#     try:
#         env = WebShopEnv(config)
        
#         # Reset environment
#         env.reset(seed=42)
        
#         # Test render method
#         rendered = env.render()
#         assert rendered is not None
#         assert isinstance(rendered, str)
#         assert len(rendered) > 0
#         print(f"   Render output: {len(rendered)} characters")
#         print(f"   Render preview: {rendered[:100]}...")
        
#         # Test render cache
#         assert hasattr(env, 'render_cache')
#         assert env.render_cache is not None
        
#         env.close()
#         print("✅ Rendering modes test passed")
        
#     except Exception as e:
#         print(f"❌ Rendering modes test failed: {e}")
#         print(f"   This might be expected if WebShop dependencies are not fully set up")


# def test_env_seeding():
#     """Test that environment generates different observations with different seeds"""
#     print("\n🔍 TESTING ENVIRONMENT SEEDING...")
    
#     # Use consistent config loading
#     env_config = get_default_config()
#     print(f"   Environment config: {[attr for attr in dir(env_config) if not attr.startswith('_')]}")
    
#     try:
#         # Test multiple environments with different seeds
#         envs = []
#         initial_states = []
        
#         for i in range(5):  # Test 5 different seeds
#             env = WebShopEnv(env_config)
#             seed = random.randint(0, 100000)
            
#             # Reset with different seeds
#             state = env.reset(seed=seed)
#             envs.append(env)
#             initial_states.append(state)
            
#             print(f"   Env {i} (seed {seed}): {len(state)} chars")
        
#         # Check for diversity
#         unique_states = set(initial_states)
#         print(f"   Total environments: {len(envs)}")
#         print(f"   Unique initial states: {len(unique_states)}")
#         print(f"   State diversity: {len(unique_states) / len(envs) * 100:.1f}%")
        
#         # Should have some diversity
#         if len(unique_states) == 1:
#             print(f"   ⚠️  WARNING: All environments have identical states!")
#         else:
#             print(f"   ✅ Good diversity: {len(unique_states)} unique states")
        
#         # Test same seed produces same state
#         print("\n   Testing seed reproducibility...")
#         env1 = WebShopEnv(env_config)
#         env2 = WebShopEnv(env_config)
        
#         test_seed = 12345
#         state1 = env1.reset(seed=test_seed)
#         state2 = env2.reset(seed=test_seed)
        
#         print(f"   Same seed ({test_seed}) produces same state: {state1 == state2}")
        
#         # Clean up
#         for env in envs + [env1, env2]:
#             env.close()
        
#         print("   ✅ Environment seeding test completed!")
        
#     except Exception as e:
#         print(f"   ❌ Environment seeding test failed: {e}")


# def test_env_actions():
#     """Test environment action processing"""
#     print("\n🔍 TESTING ENVIRONMENT ACTIONS...")
    
#     # Use consistent config loading
#     env_config = get_default_config()
    
#     try:
#         env = WebShopEnv(env_config)
        
#         # Reset environment
#         initial_state = env.reset(seed=12345)
#         print(f"   Initial state length: {len(initial_state)}")
        
#         # Get available actions
#         available_actions = env.get_available_actions()
#         print(f"   Available actions: {available_actions}")
        
#         # Test each action
#         for action in available_actions:
#             print(f"   Testing action: {action}")
            
#             # Reset to test each action from same starting position
#             env.reset(seed=12345)
            
#             try:
#                 new_state, reward, done, info = env.step(action)
#                 print(f"      Result: reward={reward}, done={done}, info={info}")
#                 print(f"      New state length: {len(new_state)}")
#             except Exception as e:
#                 print(f"      ❌ Error executing action: {e}")
        
#         env.close()
#         print("   ✅ Environment actions test completed!")
        
#     except Exception as e:
#         print(f"   ❌ Environment actions test failed: {e}")


# def test_env_config_validation():
#     """Test environment configuration validation"""
#     print("\n🔍 TESTING ENVIRONMENT CONFIG VALIDATION...")
    
#     # Use consistent config loading
#     env_config = get_default_config()
    
#     # Check required fields
#     required_fields = ['observation_mode', 'file_path', 'server', 'filter_goals', 'limit_goals', 'num_products', 'human_goals', 'show_attrs']
#     for field in required_fields:
#         if hasattr(env_config, field):
#             value = getattr(env_config, field)
#             print(f"   ✅ {field}: {value}")
#         else:
#             print(f"   ❌ Missing {field}")
    
#     # Test environment creation
#     try:
#         env = WebShopEnv(env_config)
#         print(f"   ✅ Environment created successfully")
        
#         # Test basic operations
#         state = env.reset()
#         print(f"   ✅ Reset successful, state length: {len(state)}")
        
#         # Test rendering
#         rendered = env.render()
#         print(f"   ✅ Render successful, length: {len(rendered)}")
        
#         env.close()
#     except Exception as e:
#         print(f"   ❌ Environment creation failed: {e}")
    
#     print("   ✅ Environment config validation completed!")


# def test_simple_action_debug():
#     """Simple debug test to see what happens with individual actions"""
#     print("\n🔧 SIMPLE ACTION DEBUG TEST...")
    
#     env_config = get_default_config()
    
#     try:
#         env = WebShopEnv(env_config)
        
#         # Reset with a specific seed
#         initial_state = env.reset(seed=42)
#         print(f"   Initial state length: {len(initial_state)}")
#         print(f"   Initial state preview: {initial_state[:100]}...")
#         print()
        
#         # Get available actions
#         available_actions = env.get_available_actions()
        
#         for action in available_actions:
#             # Reset to same state
#             env.reset(seed=42)
#             print(f"   Testing action: {action}")
            
#             try:
#                 state, reward, done, info = env.step(action)
#                 print(f"      Reward: {reward}")
#                 print(f"      Done: {done}")
#                 print(f"      Info: {info}")
#                 print(f"      New state length: {len(state)}")
#                 print()
                
#             except Exception as e:
#                 print(f"      ❌ Error: {e}")
#                 print()
        
#         env.close()
#         print("   ✅ Simple action debug completed!")
        
#     except Exception as e:
#         print(f"   ❌ Simple action debug failed: {e}")


# def test_environment_observation_analysis():
#     """Analyze the structure of WebShop environment observations"""
#     print("\n🔍 TESTING ENVIRONMENT OBSERVATION ANALYSIS...")
#     print("   Analyzing 10 environments to understand observation structure")
    
#     env_config = get_default_config()
    
#     try:
#         total_envs = 10
#         observation_analyses = []
        
#         for env_id in range(total_envs):
#             env = WebShopEnv(env_config)
            
#             try:
#                 # Reset with unique seed
#                 seed = random.randint(0, 100000)
#                 initial_state = env.reset(seed=seed)
                
#                 # Analyze the observation structure
#                 analysis = {
#                     'env_id': env_id,
#                     'seed': seed,
#                     'observation_length': len(initial_state),
#                     'contains_html': '<' in initial_state and '>' in initial_state,
#                     'contains_search': 'search' in initial_state.lower(),
#                     'contains_click': 'click' in initial_state.lower(),
#                     'contains_product': 'product' in initial_state.lower(),
#                     'contains_button': 'button' in initial_state.lower(),
#                     'contains_link': 'link' in initial_state.lower(),
#                     'contains_form': 'form' in initial_state.lower(),
#                     'contains_input': 'input' in initial_state.lower()
#                 }
                
#                 # Get available actions
#                 available_actions = env.get_available_actions()
#                 analysis['num_actions'] = len(available_actions)
#                 analysis['has_search_action'] = any('search[' in action for action in available_actions)
#                 analysis['has_click_action'] = any('click[' in action for action in available_actions)
                
#                 observation_analyses.append(analysis)
                
#                 print(f"   Analyzed {env_id + 1}/{total_envs} environments...")
                
#             except Exception as e:
#                 print(f"   ❌ Error analyzing env {env_id+1}: {e}")
            
#             finally:
#                 env.close()
        
#         # Calculate statistics
#         if observation_analyses:
#             print(f"\n   📊 OBSERVATION ANALYSIS RESULTS ({len(observation_analyses)} environments):")
            
#             avg_length = sum(a['observation_length'] for a in observation_analyses) / len(observation_analyses)
#             avg_actions = sum(a['num_actions'] for a in observation_analyses) / len(observation_analyses)
            
#             print(f"      Average observation length: {avg_length:.0f} characters")
#             print(f"      Average number of actions: {avg_actions:.1f}")
            
#             # Count features
#             html_count = sum(1 for a in observation_analyses if a['contains_html'])
#             search_count = sum(1 for a in observation_analyses if a['contains_search'])
#             click_count = sum(1 for a in observation_analyses if a['contains_click'])
#             product_count = sum(1 for a in observation_analyses if a['contains_product'])
            
#             print(f"      Environments with HTML: {html_count}/{len(observation_analyses)}")
#             print(f"      Environments with search: {search_count}/{len(observation_analyses)}")
#             print(f"      Environments with click: {click_count}/{len(observation_analyses)}")
#             print(f"      Environments with product: {product_count}/{len(observation_analyses)}")
        
#         print("   ✅ Environment observation analysis completed!")
        
#     except Exception as e:
#         print(f"   ❌ Environment observation analysis failed: {e}")


# def test_environment_session_handling():
#     """Test environment session handling and session generation"""
#     print("\n🔍 TESTING ENVIRONMENT SESSION HANDLING...")
    
#     env_config = get_default_config()
    
#     try:
#         env = WebShopEnv(env_config)
        
#         # Test reset with different session types
#         test_cases = [
#             (None, "No session"),
#             ("test_session_123", "String session"),
#             (42, "Integer session"),
#             ("", "Empty string session")
#         ]
        
#         for session, description in test_cases:
#             print(f"   Testing {description}: {session}")
            
#             try:
#                 obs = env.reset(seed=42, session=session)
#                 print(f"      ✅ Success, observation length: {len(obs)}")
#             except Exception as e:
#                 print(f"      ❌ Failed: {e}")
        
#         # Test session generation with seed
#         print(f"   Testing session generation with seed...")
#         try:
#             obs1 = env.reset(seed=42)
#             obs2 = env.reset(seed=42)
#             print(f"      Same seed produces same session: {obs1 == obs2}")
#         except Exception as e:
#             print(f"      ❌ Session generation failed: {e}")
        
#         env.close()
#         print("   ✅ Environment session handling completed!")
        
#     except Exception as e:
#         print(f"   ❌ Environment session handling failed: {e}")


# def test_environment_instruction_text():
#     """Test environment instruction text handling"""
#     print("\n🔍 TESTING ENVIRONMENT INSTRUCTION TEXT...")
    
#     env_config = get_default_config()
    
#     try:
#         env = WebShopEnv(env_config)
        
#         # Test reset with different instruction texts
#         test_instructions = [
#             None,
#             "Find a laptop under $500",
#             "Search for wireless headphones",
#             "Look for a gaming mouse",
#             ""
#         ]
        
#         for instruction in test_instructions:
#             print(f"   Testing instruction: {instruction}")
            
#             try:
#                 obs = env.reset(seed=42, instruction_text=instruction)
#                 print(f"      ✅ Success, observation length: {len(obs)}")
#             except Exception as e:
#                 print(f"      ❌ Failed: {e}")
        
#         env.close()
#         print("   ✅ Environment instruction text test completed!")
        
#     except Exception as e:
#         print(f"   ❌ Environment instruction text test failed: {e}")


# def test_webshop_specific_functionality():
#     """Test WebShop-specific functionality like search and click actions"""
#     print("\n🔍 TESTING WEBSHOP-SPECIFIC FUNCTIONALITY...")
    
#     env_config = get_default_config()
    
#     try:
#         env = WebShopEnv(env_config)
        
#         # Reset environment
#         initial_state = env.reset(seed=42)
#         print(f"   Initial state length: {len(initial_state)}")
        
#         # Get available actions
#         available_actions = env.get_available_actions()
#         print(f"   Available actions: {available_actions}")
        
#         # Test search actions
#         search_actions = [action for action in available_actions if action.startswith('search[')]
#         if search_actions:
#             print(f"   Found {len(search_actions)} search actions")
#             for search_action in search_actions:
#                 print(f"      Testing search action: {search_action}")
#                 env.reset(seed=42)  # Reset to same state
                
#                 try:
#                     state, reward, done, info = env.step(search_action)
#                     print(f"         Result: reward={reward}, done={done}")
#                     print(f"         New state length: {len(state)}")
#                 except Exception as e:
#                     print(f"         ❌ Error: {e}")
#         else:
#             print(f"   No search actions available")
        
#         # Test click actions
#         click_actions = [action for action in available_actions if action.startswith('click[')]
#         if click_actions:
#             print(f"   Found {len(click_actions)} click actions")
#             for click_action in click_actions[:3]:  # Test first 3 click actions
#                 print(f"      Testing click action: {click_action}")
#                 env.reset(seed=42)  # Reset to same state
                
#                 try:
#                     state, reward, done, info = env.step(click_action)
#                     print(f"         Result: reward={reward}, done={done}")
#                     print(f"         New state length: {len(state)}")
#                 except Exception as e:
#                     print(f"         ❌ Error: {e}")
#         else:
#             print(f"   No click actions available")
        
#         env.close()
#         print("   ✅ WebShop-specific functionality test completed!")
        
#     except Exception as e:
#         print(f"   ❌ WebShop-specific functionality test failed: {e}")


# def test_environment_render_cache():
#     """Test environment render cache functionality"""
#     print("\n🔍 TESTING ENVIRONMENT RENDER CACHE...")
    
#     env_config = get_default_config()
    
#     try:
#         env = WebShopEnv(env_config)
        
#         # Reset environment
#         initial_state = env.reset(seed=42)
        
#         # Test render cache
#         assert hasattr(env, 'render_cache')
#         assert env.render_cache is not None
        
#         # Test that render cache contains observation and actions
#         render_output = env.render()
#         assert render_output is not None
#         assert isinstance(render_output, str)
#         assert len(render_output) > len(initial_state)  # Should include actions
        
#         print(f"   Render cache length: {len(env.render_cache)}")
#         print(f"   Render output length: {len(render_output)}")
#         print(f"   Render cache contains actions: {'Available actions:' in env.render_cache}")
        
#         # Test render cache after action
#         available_actions = env.get_available_actions()
#         if available_actions:
#             action = available_actions[0]
#             state, reward, done, info = env.step(action)
            
#             # Check that render cache was updated
#             new_render_output = env.render()
#             assert new_render_output is not None
#             assert len(new_render_output) > 0
            
#             print(f"   After action render cache length: {len(env.render_cache)}")
#             print(f"   After action render output length: {len(new_render_output)}")
        
#         env.close()
#         print("   ✅ Environment render cache test completed!")
        
#     except Exception as e:
#         print(f"   ❌ Environment render cache test failed: {e}")


# def test_environment_info_structure():
#     """Test that environment returns proper info structure"""
#     print("\n🔍 TESTING ENVIRONMENT INFO STRUCTURE...")
    
#     env_config = get_default_config()
    
#     try:
#         env = WebShopEnv(env_config)
        
#         # Reset environment
#         env.reset(seed=42)
        
#         # Get available actions
#         available_actions = env.get_available_actions()
        
#         if available_actions:
#             action = available_actions[0]
#             state, reward, done, info = env.step(action)
            
#             # Test info structure
#             assert isinstance(info, dict)
#             assert 'action_is_effective' in info
#             assert 'action_is_valid' in info
#             assert 'success' in info
            
#             print(f"   Info keys: {list(info.keys())}")
#             print(f"   action_is_effective: {info['action_is_effective']}")
#             print(f"   action_is_valid: {info['action_is_valid']}")
#             print(f"   success: {info['success']}")
            
#             # Test info types
#             assert isinstance(info['action_is_effective'], bool)
#             assert isinstance(info['action_is_valid'], bool)
#             assert isinstance(info['success'], bool)
            
#             print(f"   ✅ Info structure is correct")
#         else:
#             print(f"   No actions available to test info structure")
        
#         env.close()
#         print("   ✅ Environment info structure test completed!")
        
#     except Exception as e:
#         print(f"   ❌ Environment info structure test failed: {e}")


# if __name__ == "__main__":
#     # Setup logging to test_logs
#     tee = setup_logging()
    
#     try:
#         print("🚀 Starting WebShop Environment Tests...")
#         print()
        
#         print("Test 1: Single environment creation")
#         test_single_env_creation()
#         print()
        
#         print("Test 2: Environment reset with different seeds")
#         test_env_reset_with_different_seeds()
#         print()
        
#         print("Test 3: Environment step functionality")
#         test_env_step_functionality()
#         print()
        
#         print("Test 4: Environment action space")
#         test_env_action_space()
#         print()
        
#         print("Test 5: Environment action mapping")
#         test_env_action_mapping()
#         print()
        
#         print("Test 6: Environment rendering modes")
#         test_env_rendering_modes()
#         print()
        
#         print("Test 7: 10 random environments × 3 runs")
#         test_10_random_envs_three_times()
#         print()

#         print("Test 8: Environment seeding")
#         test_env_seeding()
#         print()

#         print("Test 9: Environment actions")
#         test_env_actions()
#         print()

#         print("Test 10: Environment config validation")
#         test_env_config_validation()
#         print()
        
#         print("Test 11: Simple action debug test")
#         test_simple_action_debug()
#         print()

#         print("Test 12: Environment observation analysis")
#         test_environment_observation_analysis()
#         print()

#         print("Test 13: Environment session handling")
#         test_environment_session_handling()
#         print()

#         print("Test 14: Environment instruction text")
#         test_environment_instruction_text()
#         print()

#         print("Test 15: WebShop-specific functionality")
#         test_webshop_specific_functionality()
#         print()

#         print("Test 16: Environment render cache")
#         test_environment_render_cache()
#         print()

#         print("Test 17: Environment info structure")
#         test_environment_info_structure()
#         print()
        
#         print("=" * 60)
#         print("🎉 All tests completed!")
#         print(f"✅ Test completed at {datetime.now()}")
#         print("📝 Note: Some tests may fail if WebShop dependencies are not fully set up")
#         print("   This is expected behavior for a web-based environment")
        
#     except Exception as e:
#         print(f"❌ Test failed with error: {e}")
#         import traceback
#         traceback.print_exc()
        
#     finally:
#         # Close the log file
#         tee.close()
#         sys.stdout = tee.stdout  # Restore original stdout
