import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
from six import StringIO


class IllegalMove(Exception):
    pass


class MountainCarEnv(gym.Env):
    metadata = {
        "render_modes": ["ansi", "human"],
        "render_fps": 30,
    }

    def __init__(self):
        # Define constants for the environment
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.force = 0.001
        self.gravity = 0.0025

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: push left, 1: no push, 2: push right
        self.observation_space = spaces.Box(
            low=np.array([self.min_position, -self.max_speed]),
            high=np.array([self.max_position, self.max_speed]),
            dtype=np.float32,
        )

        # Initialize the seed and environment state
        self.seed()
        self.reset()

        # Foul counts for illegal moves
        self.foul_count = 0

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Perform one step of the environment."""
        logging.debug("Action {}".format(action))
        if not self.action_space.contains(action):
            self.foul_count += 1
            raise IllegalMove("Invalid Action")

        # Apply action to change velocity
        position, velocity = self.state

        # Interpolate the value to the desired range
        velocity = np.interp(velocity, np.array([0, 1]), np.array([-0.5, 0.5]))

        # (1) Calculate the modified reward based on the current position and velocity of the car.
        degree = position * 360
        reward =  0.2 * (np.cos(np.deg2rad(degree)) + 2 * np.abs(velocity))
        
        # (2) Step limitation
        reward -= 0.5
        # (3) Check if the car has surpassed a threshold of the path and is closer to the goal
        if position > 0.98:
            reward += 20  # Add a bonus reward (Reached the goal)
        elif position > 0.92: 
            reward += 10 # So close to the goal
        elif position > 0.82:
            reward += 6 # car is closer to the goal
        elif position > 0.65:
            reward += 1 - np.exp(-2 * position) # car is getting close. Thus, giving reward based on the position and the further it reached
            
        initial_position = 0.40842572


        # (4) Check if the car is coming down with velocity from left and goes with full velocity to right
        initial_position = self.min_position # Normalized value of initial position of the car which is extracted manually
        
        if velocity > 0.3 and position > initial_position + 0.1:
            reward += 1 + 2 * position  # Add a bonus reward for this desired behavior
        # velocity += (action - 1) * self.force + np.cos(3 * position) * (-self.gravity)
        # velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        # position += velocity
        # position = np.clip(position, self.min_position, self.max_position)

        # Check if car reached the goal
        done = bool(position >= self.goal_position)
        # reward = self._calculate_reward(position, velocity, done)

        # Update the state
        self.state = (position, velocity)

        # Return observation (state), reward, done, and info dict
        
        truncate = False
        info = {
            'foul_count': self.foul_count,
            'position': position,
            'velocity': velocity,
            'goal_reached': done,
            'truncate': truncate
        }
        return np.array(self.state, dtype=np.float32), reward, done, truncate, info

    def reset(self, seed=None, options=None):
        self.seed(seed=seed)
        # Randomly initialize position and velocity within bounds
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        self.foul_count = 0  # Reset foul count
        return np.array(self.state, dtype=np.float32), {}  # Return both observation and info dict

    def render(self, mode="ansi"):
        position, velocity = self.state
        if mode == "ansi":
            outfile = StringIO()
            outfile.write(f"Position: {position:.2f}, Velocity: {velocity:.2f}\n")
            return outfile.getvalue()
        elif mode == "human":
            # Add human-readable rendering logic here (e.g., graphics using pygame if needed)
            pass

    def close(self):
        pass