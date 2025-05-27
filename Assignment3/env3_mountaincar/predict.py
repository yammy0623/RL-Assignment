import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A2C, PPO, DQN

import numpy as np
from collections import Counter

register(
    id='MountainCar-eval',
    entry_point='envs:MountainCarEnv'
)

def evaluation(env, model, render_last, eval_num=100):
    """We only evaluate seeds 0-99 as our public test cases."""
    score = []
    highest = []

    ### Run eval_num times rollouts,
    for seed in range(eval_num):
        done = False
        # Set seed and reset env using Gymnasium API
        obs, info = env.reset(seed=seed)

        while not done:
            # Interact with env using Gymnasium API
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)

        # Render the last board state of each episode
        # print("Last board state:")
        # env.render()

        score.append(info['score'])
        highest.append(info['highest'])

    ### Render last rollout
    if render_last:
        print("Rendering last rollout")
        done = False
        obs, info = env.reset(seed=eval_num-1)
        env.render()

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            env.render()

        
    return score, highest


if __name__ == "__main__":
    model_path = "models/sample_model/66"  # Change path name to load different models
    env = gym.make('MountainCar-eval')

    ### Load model with SB3
    # Note: Model can be loaded with arbitrary algorithm class for evaluation
    # (You don't necessarily need to use PPO for training)
    # model = PPO.load(model_path)
    # model = A2C.load(model_path)
    max_episodes = 1000
    eval_num = 100
    model = DQN.load(model_path)
    model.eval()
    frames = []

    # Testing loop over episodes
    for episode in range(1, max_episodes+1):         
        obs, info = env.reset(seed=eval_num-1)
        env.render()
        truncation = False
        step_size = 0
        episode_reward = 0
                                                        
        while not done and not truncation:
            
            action, next_state = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            # Capture the current state as an image for GIF
            env.render()           
            state = next_state
            episode_reward += reward
            step_size += 1
                                                                                                                
        # Print log            
        result = (f"Episode: {episode}, "
                    f"Steps: {step_size:}, "
                    f"Reward: {episode_reward:.2f}, ")
        print(result)
        # Create GIF from collected frames
        # gif_filename = "test_episode.gif"
        # imageio.mimsave(gif_filename, frames, duration=0.1)  # Adjust duration to control speed
        # print(f"GIF saved as {gif_filename}")

    
