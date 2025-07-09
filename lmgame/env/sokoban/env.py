import gym
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
import numpy as np
from .utils import generate_room
from lmgame.env.base import BaseDiscreteActionEnv
from lmgame.utils import all_seed
import ray

class SokobanEnv(BaseDiscreteActionEnv, GymSokobanEnv):
    def __init__(self, config, **kwargs):
        self.config = config
        self.GRID_LOOKUP = self.config['grid_lookup']
        self.ACTION_LOOKUP = self.config['action_lookup']
        self.search_depth = self.config['search_depth']
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)
        self.render_mode = self.config['render_mode']

        BaseDiscreteActionEnv.__init__(self)
        GymSokobanEnv.__init__(
            self,
            dim_room=self.config['dim_room'], 
            max_steps=self.config['max_steps'],
            num_boxes=self.config['num_boxes'],
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
        info = {'action_is_valid': True}
        previous_pos = self.player_position
        _, reward, done, _ = GymSokobanEnv.step(self, action) 
        next_obs = self.render()
        info["action_is_effective"] = not np.array_equal(previous_pos, self.player_position)
        info["success"] = self.boxes_on_target == self.num_boxes
        return next_obs, reward, done, info

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

@ray.remote(num_cpus=0.1)
class SokobanEnvActor:
    def __init__(self, config):
        self.env = SokobanEnv(config)

    def reset(self, seed=None):
        return self.env.reset(seed)

    def step(self, actions):
        obs = self.env.render()
        total_reward = 0
        info = {'action_is_valid': False, 'action_is_effective': False, 'success': False}
        done = False
        executed_actions = []
        for action in actions:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            executed_actions.append(action)
            if done:
                break
        return obs, total_reward, done, info, executed_actions

    def get_config(self):
        return self.env.config

    def render(self, mode=None):
        return self.env.render(mode)

    def close(self):
        self.env.close()

# python -m lmgame.env.sokoban.env
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    config = {
        "dim_room": (6, 6),
        "num_boxes": 1,
        "max_steps": 10,
        "search_depth": 5,
        "grid_lookup": {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"},
        "grid_vocab": {"#": "wall", "_": "empty", "O": "target", "√": "box on target", "X": "box", "P": "player", "S": "player on target"},
        "action_lookup": {1: "Up", 2: "Down", 3: "Left", 4: "Right"},
        "render_mode": "text"
    }
    env = SokobanEnvActor.remote(config)
    for i in range(1):
        print(ray.get(env.reset.remote(seed=1010 + i)))
        print()
    while True:
        keyboard = input("Enter action: ")
        if keyboard == 'q':
            break
        action = int(keyboard)
        obs, reward, done, info, executed_actions = ray.get(env.step.remote([action]))
        print(f"Obs:\n{obs}")
        print(f"Reward: {reward}, Done: {done}, Info: {info}, Executed: {executed_actions}")
    np_img = ray.get(env.render.remote('rgb_array'))
    # save the image
    plt.imsave('sokoban1.png', np_img)