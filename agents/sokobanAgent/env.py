import gym
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
import numpy as np
from agents.agent_utils import all_seed
from .utils import generate_room

class SokobanEnv(GymSokobanEnv):
    def __init__(self, config, **kwargs):
        self.config = config
        self.GRID_LOOKUP = self.config.get('grid_lookup', {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"})
        self.ACTION_LOOKUP = self.config.get('action_lookup', {1: "Up", 2: "Down", 3: "Left", 4: "Right"})
        self.search_depth = self.config.get('search_depth', 300)
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)  # Our config uses actions 1-4
        self.render_mode = self.config.get('render_mode', 'text')

        GymSokobanEnv.__init__(
            self,
            dim_room=self.config.get('dim_room', (6, 6)), 
            max_steps=self.config.get('max_steps', 100),
            num_boxes=self.config.get('num_boxes', 1),
            **kwargs
        )

    def reset(self, seed=None):
        try:
            with all_seed(seed):
                self.room_fixed, self.room_state, self.box_mapping, action_sequence = generate_room(
                    dim=self.dim_room,
                    num_steps=self.num_gen_steps,
                    num_boxes=self.num_boxes,
                    search_depth=self.search_depth
                )
            self.num_env_steps, self.reward_last, self.boxes_on_target = 0, 0, 0
            self.player_position = np.argwhere(self.room_state == 5)[0]
            return self.render()
        except (RuntimeError, RuntimeWarning) as e:
            next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
            return self.reset(next_seed)
        
    def step(self, action: int):
        """
        Take an action in the environment.
        Actions: 1=Up, 2=Down, 3=Left, 4=Right (from our config)
        """
        previous_pos = self.player_position.copy()
        
        # ✅ FAULT TOLERANCE: Check if action is valid before processing
        if action not in self.ACTION_LOOKUP:
            # Invalid action - return current state with penalty and mark as invalid
            info = {
                'action_is_valid': False,
                'action_is_effective': False,
                'success': self.boxes_on_target == self.num_boxes
            }
            next_obs = self.render()
            return next_obs, -0.1, False, info  # Small penalty for invalid action
        
        # Use our direct step logic instead of gym_sokoban
        try:
            # Map our action to internal coordinates
            # Our config: 1=Up, 2=Down, 3=Left, 4=Right
            # Internal: 0=Up, 1=Down, 2=Left, 3=Right
            action_internal = action - 1
            
            # Get movement direction
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
            if action_internal < 0 or action_internal >= len(moves):
                # Invalid action
                info = {
                    'action_is_valid': False,
                    'action_is_effective': False,
                    'success': self.boxes_on_target == self.num_boxes
                }
                return self.render(), -0.1, False, info
            
            move = moves[action_internal]
            new_player_pos = (self.player_position[0] + move[0], self.player_position[1] + move[1])
            
            reward = 0
            done = False
            
            # Check boundaries
            if (new_player_pos[0] < 0 or new_player_pos[0] >= self.room_state.shape[0] or
                new_player_pos[1] < 0 or new_player_pos[1] >= self.room_state.shape[1]):
                # Out of bounds - no movement
                info = {
                    'action_is_valid': True,
                    'action_is_effective': False,
                    'success': self.boxes_on_target == self.num_boxes
                }
                return self.render(), -0.01, False, info
            
            # Check if hitting wall
            if self.room_fixed[new_player_pos] == 0:  # Wall
                # Can't move into wall
                info = {
                    'action_is_valid': True,
                    'action_is_effective': False,
                    'success': self.boxes_on_target == self.num_boxes
                }
                return self.render(), -0.01, False, info
            
            # Check if there's a box in new position
            if self.room_state[new_player_pos] in [3, 4]:  # Box on target or box not on target
                # Try to push the box
                box_new_pos = (new_player_pos[0] + move[0], new_player_pos[1] + move[1])
                
                # Check if box can be pushed
                if (box_new_pos[0] < 0 or box_new_pos[0] >= self.room_state.shape[0] or
                    box_new_pos[1] < 0 or box_new_pos[1] >= self.room_state.shape[1] or
                    self.room_fixed[box_new_pos] == 0 or  # Wall
                    self.room_state[box_new_pos] in [3, 4]):  # Another box
                    # Can't push box
                    info = {
                        'action_is_valid': True,
                        'action_is_effective': False,
                        'success': self.boxes_on_target == self.num_boxes
                    }
                    return self.render(), -0.01, False, info
                
                # Push the box
                # Remove box from current position
                self.room_state[new_player_pos] = self.room_fixed[new_player_pos]
                
                # Place box in new position
                if self.room_fixed[box_new_pos] == 2:  # Target
                    self.room_state[box_new_pos] = 3  # Box on target
                    reward += 1.0  # Reward for placing box on target
                else:
                    self.room_state[box_new_pos] = 4  # Box not on target
                    reward -= 0.1  # Small penalty for moving box off target if it was on one
            
            # Move player
            self.room_state[self.player_position[0], self.player_position[1]] = self.room_fixed[self.player_position[0], self.player_position[1]]
            self.room_state[new_player_pos] = 5
            self.player_position = np.array(new_player_pos)
            
            # Count boxes on targets
            self.boxes_on_target = np.sum(self.room_state == 3)
            
            # Check if puzzle is solved
            success = self.boxes_on_target == self.num_boxes
            if success:
                reward += 10.0  # Big reward for solving
                done = True
            
            # Check if max steps reached
            self.num_env_steps += 1
            if self.num_env_steps >= self.max_steps:
                done = True
                if not success:
                    reward -= 1.0  # Penalty for not solving in time
            
            # Small step penalty to encourage efficiency
            reward -= 0.01
            
            info = {
                'action_is_valid': True,
                'action_is_effective': not np.array_equal(previous_pos, self.player_position),
                'success': success
            }
            
            next_obs = self.render()
            return next_obs, reward, done, info
            
        except Exception as e:
            # ✅ FAULT TOLERANCE: Handle any other step errors gracefully
            print(f"Warning: Environment step failed for action {action}: {e}")
            info = {
                'action_is_valid': False,
                'action_is_effective': False,
                'success': self.boxes_on_target == self.num_boxes
            }
            next_obs = self.render()
            return next_obs, -0.1, False, info  # Small penalty for failed action

    def render(self, mode=None):
        render_mode = mode if mode is not None else self.render_mode
        if render_mode == 'text':
            room = np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
            return '\n'.join(''.join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in room.tolist())
        elif render_mode == 'rgb_array':
            return self.get_image(mode='rgb_array', scale=1)
        else:
            raise ValueError(f"Invalid mode: {render_mode}")
    
    def get_all_actions(self):
        return list([k for k in self.ACTION_LOOKUP.keys()])
    
    def close(self):
        self.render_cache = None
        super(SokobanEnv, self).close()
