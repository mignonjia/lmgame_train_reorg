import gym
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
import numpy as np
import random
from .utils import generate_room
from ragen.env.base import BaseDiscreteActionEnv
from ragen.env.sokoban.config import SokobanEnvConfig
from ragen.utils import all_seed

# Import shared core logic for consistency
try:
    from gamingagent.envs.custom_02_sokoban.sokobanCore import SokobanCore, ROOM_STATE_TO_CHAR
    USING_SHARED_CORE = True
except ImportError:
    # Fallback if core not available
    USING_SHARED_CORE = False
    ROOM_STATE_TO_CHAR = {0: '#', 1: ' ', 2: '?', 3: '*', 4: '$', 5: '@', 6: '+'}

class SokobanEnv(BaseDiscreteActionEnv, GymSokobanEnv):
    def __init__(self, config=None, **kwargs):
        self.config = config or SokobanEnvConfig()
        self.GRID_LOOKUP = self.config.grid_lookup
        self.ACTION_LOOKUP = self.config.action_lookup
        self.search_depth = self.config.search_depth
        
        # Initialize shared core for consistent game logic
        if USING_SHARED_CORE:
            self.core = SokobanCore()
        else:
            self.core = None
            
        # Action space aligned with shared core (5 actions: 0=no-op, 1-4=directions)
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(5)
        self.render_mode = self.config.render_mode

        BaseDiscreteActionEnv.__init__(self)
        GymSokobanEnv.__init__(
            self,
            dim_room=self.config.dim_room, 
            max_steps=self.config.max_steps,
            num_boxes=self.config.num_boxes,
            **kwargs
        )

    def reset(self, seed=None):
        # Use a fresh random seed each time unless a specific seed is provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            
        try:
            with all_seed(seed):
                self.room_fixed, self.room_state, self.box_mapping, action_sequence = generate_room(
                    dim=self.dim_room,
                    num_steps=self.num_gen_steps,
                    num_boxes=self.num_boxes,
                    search_depth=self.search_depth
                )
                
            # Set up shared core if available
            if self.core:
                self.core.set_room_state(self.room_fixed, self.room_state)
            else:
                # Fallback: set player position manually
                self.player_position = np.argwhere(self.room_state == 5)[0]
                
            self.num_env_steps, self.reward_last, self.boxes_on_target = 0, 0, 0
            return self.render()
        except (RuntimeError, RuntimeWarning) as e:
            next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
            return self.reset(next_seed)
        
    def step(self, action: int):
        previous_pos = self.player_position if not self.core else self.core.player_position
        
        if self.core:
            # Use shared core logic for consistent movement
            moved_player, moved_box = self.core.move_or_push(action)
            reward = self.core.calculate_reward()
            done = self.core.check_all_boxes_on_target()
            
            # Update gym_sokoban state to match core state
            if hasattr(self, 'room_state'):
                self.room_state = self.core.room_state.copy()
            if hasattr(self, 'player_position'):
                self.player_position = self.core.player_position.copy() if self.core.player_position is not None else None
                
        else:
            # Fallback to original gym_sokoban logic
            _, reward, done, _ = GymSokobanEnv.step(self, action)
            moved_player = not np.array_equal(previous_pos, self.player_position)
            moved_box = False  # gym_sokoban doesn't track this separately
            
        next_obs = self.render()
        info = {
            "action_is_effective": moved_player or moved_box, 
            "action_is_valid": True, 
            "success": done
        }
            
        return next_obs, reward, done, info

    def render(self, mode=None):
        render_mode = mode if mode is not None else self.render_mode
        if render_mode == 'text':
            if self.core:
                return self.core.get_room_as_text()
            else:
                # Fallback rendering
                room = np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
                return '\n'.join(''.join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in room.tolist())
        elif render_mode == 'rgb_array':
            return self.get_image(mode='rgb_array', scale=1)
        else:
            raise ValueError(f"Invalid mode: {render_mode}")
    
    def get_all_actions(self):
        return list(self.ACTION_LOOKUP.keys()) if self.ACTION_LOOKUP else [0, 1, 2, 3, 4]
    
    def close(self):
        self.render_cache = None
        super(SokobanEnv, self).close()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    config = SokobanEnvConfig(dim_room=(6, 6), num_boxes=1, max_steps=10, search_depth=5)
    env = SokobanEnv(config)
    for i in range(10):
        print(env.reset(seed=1010 + i))
        print()
    while True:
        keyboard = input("Enter action: ")
        if keyboard == 'q':
            break
        action = int(keyboard)
        valid_actions = env.get_all_actions()
        assert action in valid_actions, f"Invalid action: {action}. Valid actions: {valid_actions}"
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
    np_img = env.get_image('rgb_array')
    # save the image
    plt.imsave('sokoban1.png', np_img)