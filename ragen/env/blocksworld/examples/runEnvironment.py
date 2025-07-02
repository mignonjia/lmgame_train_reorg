#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 13:01:41 2019

@author: rgarzon
Modified to work with newer gym versions
"""
import random
import gym
import numpy as np

# Create environment
env = gym.make('gym_blocksworld:BlocksWorld-v0')

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)

numBlocks = 3
num_episodes = 1

ep_lengths = []
for episode in range(num_episodes):
    # Reset environment
    obs, info = env.reset(seed=seed)
    done = False
    steps = 0
    
    for steps in range(40):
        # Random action
        action = [random.randint(1, numBlocks), random.randint(0, numBlocks)]
        
        # Take step
        
        print(f'State: {env.state}')
        print(f'Action: {action}')
        print(f'Goal: {env.goal}')

        obs, reward, done, info = env.step(action)
        print(f'Info: {info}')
        # print(f'Observation: {obs}')
        # print(f'Reward: {reward}')
        # print(f'Done: {done}')
        if done:
            break
        
        # Optional: render environment
        # env.render()
    
    ep_lengths.append(steps)
    print(f'\nEpisode {episode + 1} finished in {steps} steps')
    print('----------------')

print(f'Number of blocks: {numBlocks}')
print(f'Number of episodes run: {num_episodes}')
print(f'Average episode length: {sum(ep_lengths) / len(ep_lengths):.2f} steps')
