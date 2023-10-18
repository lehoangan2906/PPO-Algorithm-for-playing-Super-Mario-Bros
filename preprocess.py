import gym
from torchvision import transforms
from gym.spaces import Box
import numpy as np
import torch

# Frame skipping
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False

        # Iterate over frames with skipping
        for _ in range(self._skip):
            # Accumulate reward and check if the game is done
            obs, reward, done, trunc, info = self.env.step(action) 
            truncated = truncated or trunc  # If the game is truncated, it means that Mario died
            total_reward += reward        # Accumulate reward
            
            if done:
                break

        return obs, total_reward, done, truncated, info
    

# Make frame grayscale
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Set observation space to grayscale
        self.observation_space = Box(low = 0, high = 255, shape = self.observation_space.shape[:2], dtype = np.uint8)

    # Convert frame to grayscale
    def observation(self, observation):
        # Apply grayscale filter
        transform = transforms.Grayscale()
        return transform(torch.tensor(np.transpose(observation, (2, 0, 1)).copy(), dtype = torch.float)) # Convert to torch tensor and apply grayscale filter
    

# Resize frame
class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        # Set target shape for observation
        self.shape = (shape, shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        # Adjust observation space
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        # Apply resize and normalization transformations
        transformations = transforms.Compose([transforms.Resize(self.shape), transforms.Normalize(0, 255)])
        return transformations(observation).squeeze(0)