import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random

# Add the project root to the path to import lmgame modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from lmgame.env.sokoban.env import SokobanEnv
from lmgame.env.sokoban.config import SokobanEnvConfig

def main():
    # Create sokoban_images folder in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, 'sokoban_images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Create Sokoban environment
    config = SokobanEnvConfig(
        dim_room=(6, 6), 
        num_boxes=2, 
        max_steps=50, 
        search_depth=5,
        render_mode='rgb_array'
    )
    env = SokobanEnv(config)
    
    # Run 2 episodes
    for episode_id in range(2):
        print(f"Starting episode {episode_id}")
        
        # Reset environment for new episode
        obs = env.reset(seed=random.randint(0, 10000))
        
        # Save initial state (image 0)
        initial_image = env.render('rgb_array')
        image_filename = f"sokoban_{episode_id:03d}_{0:03d}.png"
        image_path = os.path.join(images_dir, image_filename)
        plt.imsave(image_path, initial_image)
        print(f"Saved initial state: {image_filename}")
        
        # Run 4 more random actions (to get 5 total images per episode)
        valid_actions = env.get_all_actions()
        
        for step in range(4):
            # Choose random action
            action = random.choice(valid_actions)
            print(f"Episode {episode_id}, Step {step+1}: Taking action {action}")
            
            # Take action
            obs, reward, done, info = env.step(action)
            
            # Get RGB image
            rgb_image = env.render('rgb_array')
            
            # Save image with specified naming convention
            image_filename = f"sokoban_{episode_id:03d}_{step+1:03d}.png"
            image_path = os.path.join(images_dir, image_filename)
            plt.imsave(image_path, rgb_image)
            print(f"Saved image: {image_filename} (reward: {reward}, done: {done})")
            
            if done:
                print(f"Episode {episode_id} finished early at step {step+1}")
                break
        
        print(f"Completed episode {episode_id}\n")
    
    env.close()
    print(f"All images saved to: {images_dir}")
    
    # List all saved images
    saved_images = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    print(f"Saved {len(saved_images)} images:")
    for img in saved_images:
        print(f"  {img}")

if __name__ == '__main__':
    main()
