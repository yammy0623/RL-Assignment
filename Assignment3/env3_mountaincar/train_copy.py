import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

for model_name in [DQN, PPO]:
    # Step 1: Create the environment
    env = gym.make("MountainCar-v0", render_mode="human")

    # Step 2: Initialize the model
    model = model_name("MlpPolicy", env, verbose=1, tensorboard_log=f"./{str(model_name)}_mountaincar_tensorboard/")

    # Step 3: Train the model
    model.learn(total_timesteps=100000)

    # Step 4: Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

    # Step 5: Save the model
    model.save("dqn_mountaincar")

    # Step 6: Load the model (optional)
    # model = DQN.load("dqn_mountaincar", env=env)

    # Step 7: Run the trained agent
    obs, _ = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        env.render()

    env.close()
