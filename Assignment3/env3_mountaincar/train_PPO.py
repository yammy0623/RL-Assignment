import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DQN, PPO, SAC


warnings.filterwarnings("ignore")
# register(
#     id='MountainCar-v0',
#     entry_point='envs:MountainCarEnv'
# )

# Set hyper params (configurations) for training
my_config = {
    "run_id": "car",

    "algorithm": PPO,
    "policy_network": "MlpPolicy",
    "save_path": "models/sample_model",

    "epoch_num": 100,
    "timesteps_per_epoch": 10000,
    "eval_episode_num": 10,
    "learning_rate": 1e-3,
}


def make_env():
    env = gym.make('MountainCar-v0')
    return env

def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    avg_score = 0
    avg_highest = -1.2
    for seed in range(eval_episode_num):
        done = False
        truncated = False
        env.seed(seed)
        obs = env.reset()
        episode_score = 0
        highest_position = -float("inf")

        # Interact with the environment
        while not (done or truncated):  # Check for both done and truncated
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)  # Updated step syntax
            truncated = info[0].get('truncate', False) if info else False
            episode_score += reward
            highest_position = max(highest_position, obs[0, 0])  # Remove [0,0] indexing since obs is already flat

        # Accumulate scores for averaging
        avg_highest += highest_position
        avg_score += episode_score

    # Compute average values
    avg_highest /= eval_episode_num
    avg_score /= eval_episode_num

    return avg_score, avg_highest

def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best = 0
    for epoch in range(config["epoch_num"]):

        # Uncomment to enable wandb logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            # callback=WandbCallback(
            #     gradient_save_freq=100,
            #     verbose=2,
            # ),
        )

        ### Evaluation
        # print(config["run_id"])
        # print("Epoch: ", epoch)
        avg_score, avg_highest = eval(eval_env, model, config["eval_episode_num"])
        
        # print("Avg_score:  ", avg_score)
        # print("Avg_highest:", avg_highest)
        print()
        wandb.log(
            {"epoch": epoch,
             "position": avg_highest,
             "reward": avg_score
             }
        )
        

        ### Save best model
        print(avg_highest)
        if current_best < avg_highest:
            print("Saving Model")
            current_best = avg_highest
            save_path = config["save_path"]
            model.save(f"{save_path}/DQN/{epoch}")
            print(epoch)
        # print("---------------")


if __name__ == "__main__":

    # Create wandb session (Uncomment to enable wandb logging)
    run = wandb.init(
        project="rl_hw3_MountainCar_PPO",
        config=my_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        id=my_config["run_id"],
        name="PPO"
    )

    # Create training environment 
    num_train_envs = 2
    train_env = DummyVecEnv([make_env for _ in range(num_train_envs)])

    # Create evaluation environment 
    eval_env = DummyVecEnv([make_env])  

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = my_config["algorithm"](
        my_config["policy_network"], 
        train_env, 
        verbose=2,
        tensorboard_log=my_config["run_id"],
        learning_rate=my_config["learning_rate"],
    )

    train(eval_env, model, my_config)
    # wandb.finish()