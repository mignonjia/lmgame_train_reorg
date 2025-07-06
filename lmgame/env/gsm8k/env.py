import gym
from gym import spaces
import numpy as np
from datasets import load_dataset
import re
import random
from lmgame.env.base import BaseLanguageBasedEnv
from lmgame.utils import all_seed
from .config import GSM8KEnvConfig
from collections import defaultdict
from openai import OpenAI
import os
import json

class GSM8KEnv(BaseLanguageBasedEnv):
    def __init__(self, config: GSM8KEnvConfig):
        super(GSM8KEnv, self).__init__()
        
        self.config = config
        self.dataset = load_dataset(self.config.dataset_path, 'main', split=self.config.split)
        self.current_sample = None
        self.current_unique_id = None
        self.current_question = None
        self.correct_answer = None
        self.step_num = None
        self.render_cache = None

    def extract_answer(self, answer):
        if '####' in answer:
            answer = answer.split('####')[-1].strip()
        else:
            answer = answer.strip()

        for remove_char in [',', '$', '%', 'g']:
            answer = answer.replace(remove_char, '')

        try:
            return int(answer)
        except ValueError:
            return answer


    def reset(self,seed=None):
        # print("[DEBUG] seed ", seed)
        with all_seed(seed):
            question_data = random.choice(self.dataset)

        self.current_sample = question_data
        self.current_question = question_data['question']
        self.correct_solution = question_data['answer']
        self.correct_answer = self.extract_answer(question_data['answer'])
        self.render_cache = self.current_question
        self.step_num = 0
        return self.render_cache
        
    def step(self, action):
        is_correct, is_valid = self._check_answer(action)
        # reward = 1.0 / (2 ** self.step_num) if is_correct else 0.0
        reward = 10.0 if is_correct else -0.1
        if is_correct:
            observation = "Correct!"
            done = True
        else:
            observation = "Incorrect. Please think again."
            done = False
        self.step_num += 1
        info = {"action_is_effective": True, "action_is_valid": is_valid, "success": is_correct}
        self.render_cache = observation
        return self.render_cache, reward, done, info
    
    def _check_answer(self, user_answer):
        # print("[DEBUG] question:", self.current_question)
        # print("[DEBUG] correct_answer:", self.correct_answer)
        # print("[DEBUG] user_answer:", user_answer)
        """Check if the user's answer matches the correct answer."""
        user_answer = user_answer.strip()
        matches = re.findall(r'\d+', user_answer)
        last_integer = int(matches[-1]) if matches else None
        # print("[DEBUG] user_answer_integer:", last_integer)

        is_correct = last_integer == self.correct_answer
        is_valid = last_integer != ""
        return is_correct, is_valid

    def render(self):
        return self.render_cache


if __name__ == "__main__":
    # Create the environment configuration
    config = GSM8KEnvConfig(
        dataset_path="openai/gsm8k",
        split="train",
        max_steps=10,
    )
    
    # Initialize the environment
    env = GSM8KEnv(config)
    
    # Reset the environment to get the first question
    print("Question:")
    question = env.reset(seed=42)
    print(question)
    print("\nCorrect answer (for testing purposes):")
    print(env.correct_answer)
    
    # Interactive loop for testing
    while True:
        user_answer = input("\nEnter your answer (or 'q' to quit): ")
        if user_answer.lower() == 'q':
            break
        
        # Take a step in the environment with the user's answer
        obs, reward, done, info = env.step(user_answer)
        
        
        # Print the results
        print("\nFeedback:", obs)
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)
        
        # If the episode is done, reset the environment for a new question
        if done:
            print("\n--- New Question ---")
            question = env.reset()
            print(question)
            print("\nCorrect answer (for testing purposes):")
            print(env.correct_answer)