import gym
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from preprocess import SkipFrame, GrayScaleObservation, ResizeObservation
from agent import Agent
from utils import load_checkpoint
import os


# Directory to save the models
saves_directory = "./model"


# Hyperparameters for tuning
gamma = 0.95
lamda = 0.95
batch_size = 40960
divisor = 4
epilon = 0.2
epochs = 30
a_lr= 0.00025
c_lr = 0.001
interval = 10    # Save interval (saves model and writes to csv every x episodes)


# Rendering options (True if you want to see the game being played, but might be computationally expensive)
show_game = True


# Run modes
train = True   # Set to false if you just want to run the current policy without updating it
start = 5000    # Set to 0 if you want to start from scratch, otherwise set to the episode you want to start from
verbose = True     # Show output in console


# Setup the environment for pre-processing
if show_game:
    env = gym.make('SuperMarioBros-1-1-v0', apply_api_compatibility = True, render_mode = 'human')
else:
    env = gym.make('SuperMarioBros-1-1-v0', apply_api_compatibility = True)


# Wrap the environment with the pre-processing functions
env = JoypadSpace(env, [['right'],['right', 'A']])  # only allow right and jump-right actions
env = SkipFrame(env, skip = 4)  # skip 4 frames
env = GrayScaleObservation(env)  # convert the observation to grayscale
env = ResizeObservation(env, shape = 84)  # resize the observation to 84x84
env = FrameStack(env, num_stack = 4)    # stack 4 frames


# Create PPO agent
PPO_agent = Agent(env, saves_directory, gamma, lamda, epilon, epochs, divisor, interval, batch_size, a_lr, c_lr, show_game, verbose)


# Check if saves directory exists, if not create it and start from scratch, else load the latest checkpoint
load_checkpoint(saves_directory, PPO_agent, start)


# Continue training the agent
if train:
    while True:
        PPO_agent.train(PPO_agent.sample())
else:
    while True:
        PPO_agent.sample()