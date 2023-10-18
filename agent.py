import torch
import numpy as np
from torch import nn
from model import Model
from utils import save_checkpoint, write_to_csv


# Set device as "mps" (Mac Silicon)
type = "mps"
device = torch.device(type)

class Agent:
    def __init__(self, env, save_directory, gamma, lamda, epilon, epochs, divisor, interval, batch_size, a_lr, c_lr, show, verbose):
        """
        Initializes the reinforcement learning agent.

        Parameters:
            - env: The environment for the agent to interact with.
            - save_directory: The directory to save models and training data.
            - gamma: Discount factor for future rewards.
            - lamda: Lambda parameter for Generalized Advantage Estimation (GAE).
            - epilson: Clip range for the PPO algorithm.
            - epochs: Number of optimization epochs per training iteration.
            - divisor: Divisor for mini-batch size calculation.
            - interval: Save interval (saves model and writes to csv every x episodes).
            - batch_size: Total batch size for sampling.
            - a_lr: Learning rate for the actor network.
            - c_lr: Learning rate for the critic network.
            - show: Whether to display the game during training.
            - verbose: Whether to print training details.
        """
        self.env = env
        self.show = show
        self.rewards = []   # List to store rewards for each episode
        self.total_rewards = []   # List to store total rewards for each episode
        self.episode = 0    # Episode counter
        self.gamma, self.lamda, self.epilon, self.epochs, self.interval, self.batch_size, self.save_directory, self.verbose = gamma, lamda, epilon, epochs, interval, batch_size, save_directory, verbose
        self.mini_batch_size = self.batch_size // divisor  # Calculate mini-batch size
        self.obs = np.array(self.env.reset()[0])  # Reset the environment and get the initial observation

        # Create current policy and clone it to old policy 
        self.model = Model(env).to(device) # Create the model
        self.model_old = Model(env).to(device) # Create the old model
        self.model_old.load_state_dict(self.model.state_dict()) # Clone the current model to the old model
        
        # Create optimizer for the model
        self.optimizer = torch.optim.Adam([
            {'params': self.model.actor.parameters(), 'lr': a_lr},
            {'params': self.model.critic.parameters(), 'lr': c_lr}
        ], eps=1e-4)    

        
        self.mse_loss = nn.MSELoss()    # Create MSE loss function

    
    def sample(self):
        # Initialize empty numpy arrays for sampling
        rewards = np.zeros(self.batch_size, dtype=np.float32)
        actions = np.zeros(self.batch_size, dtype=np.float32)
        log_probs = np.zeros(self.batch_size, dtype=np.float32)
        values = np.zeros(self.batch_size, dtype=np.float32)
        done = np.zeros(self.batch_size, dtype=np.bool_)
        obs = np.zeros((self.batch_size, 4, 84, 84), dtype=np.float32)
        for time in range(self.batch_size):
            # Sample actions
            with torch.no_grad():
                obs[time] = self.obs
                # Forward pass through networks
                policy, value = self.model_old(torch.tensor(self.obs, dtype=torch.float32, device=device).unsqueeze(0))
                # Get value at time
                values[time] = value.cpu().numpy()
                # Get action probability distribution
                action = policy.sample()
                # Get action from policy
                actions[time] = action.cpu().numpy()
                # Get log probability
                log_probs[time] = policy.log_prob(action).cpu().numpy()
            self.obs, rewards[time], done[time], _, _ = self.env.step(actions[time])
            self.obs = self.obs.__array__()

            if self.show:
                self.env.render()
            
            if self.verbose:
                print("Episode: {}, Step: {}, reward: {}".format(self.episode, time, rewards[time]))
            
            self.rewards.append(rewards[time])
            if done[time]:
                # If done increment episodes and reset environment
                self.episode += 1
                self.total_rewards.append(np.sum(self.rewards))
                self.rewards = []
                self.env.reset()
                if self.episode % self.interval == 0:
                    print('Episode: {}, average reward: {}'.format(self.episode, np.mean(self.total_rewards[-10:])))
                    write_to_csv(self.save_directory, 'training_data.csv', self.episode, np.mean(self.total_rewards[-10:]))
                    save_checkpoint(self.model_old, self.episode, self.save_directory)
        # Get advantages
        gae, advantages = self.calculate_advantages(done, rewards, values)
        return {
            'obs': torch.tensor(obs.reshape(obs.shape[0], *obs.shape[1:]), dtype=torch.float32, device=device),
            'actions': torch.tensor(actions, device=device),
            'values': torch.tensor(values, device=device),
            'log_probs': torch.tensor(log_probs, device=device),
            'advantages': torch.tensor(advantages, device=device, dtype=torch.float32),
            'gae': torch.tensor(gae, device=device, dtype=torch.float32)
        }



    def calculate_advantages(self, done, rewards, values):
        """
        Calculates Generalized Advantage Estimation (GAE) advantages.

        Parameters:
        - done: Array indicating whether each step is the end of an episode.
        - rewards: Array of rewards obtained at each step.
        - values: Array of value estimates obtained at each step.

        Returns:
        Tuple containing GAE returns and normalized advantages.
        """

        _, value = self.model_old(torch.tensor(self.obs, dtype = torch.float32, device = device).unsqueeze(0)) # Get the value of the last observation
        value = value.detach().cpu().numpy()  # Convert the value to a numpy array
        values = np.append(values, value) # Append the value of the last observation to the values array

        returns = []
        gae = 0

        for i in reversed(range(len(rewards))):
            mask = 1.0 - done[i] # Calculate the mask for the current step
            delta = rewards[i] + self.gamma * values[i + 1] * mask - values[i] # Calculate the TD error for the current step
            gae = delta + self.gamma * self.lamda * mask * gae # Calculate the GAE for the current step
            returns.insert(0, gae + values[i]) # Insert the GAE return for the current step to the beginning of the returns list

        adv = np.array(returns) - values[: -1] # Calculate the advantages

        # return the GAE returns and normalized advantages
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

    def calculate_loss(self, epilon, samples):
        """
        Calculates the loss for the PPO algorithm.

        Parameters:
        - epilson: Clip range for the PPO algorithm.
        - samples: Dictionary containing sampled data.

        Returns:
        The calculated loss.
        """

        gae = samples['gae'] # Get the GAE returns
        sampled_advantages = samples['advantages'] # Get the advantages
        policy, value = self.model(samples['obs']) # Get the policy and value from the current model
        ratio = torch.exp(policy.log_prob(samples['actions']) - samples['log_probs'])    # Calculate the ratio between the current policy and the old policy
        clipped_ratio = torch.clamp(ratio, 1.0 - epilon, 1.0 + epilon)    # Clip the ratio to make sure it is within the clip range
        clipped_objective = torch.min(ratio * sampled_advantages, clipped_ratio * sampled_advantages)    # Calculate the clipped objective which is the minimum between the ratio * advantages and the clipped ratio * advantages

        s = policy.entropy() # Calculate the entropy of the policy
        vf_loss = self.mse_loss(value, gae) # Calculate the value function loss
        loss = -clipped_objective + 0.5 * vf_loss - 0.01 * s # Calculate the total loss

        return loss.mean()


    def train(self, samples):
        """
        Trains the agent using the collected samples.

        Parameters:
        - samples: Dictionary containing sampled data.
        """

        indexes = torch.randperm(self.batch_size) # Generate random indexes for shuffling the data

        for start in range(0, self.batch_size, self.mini_batch_size):
            mini_batch_indexes = indexes[start: start + self.mini_batch_size] # Get the indexes for the current mini-batch
            mini_batch = {}

            for i, v in samples.items():
                mini_batch[i] = v[mini_batch_indexes] # Get the data for the current mini-batch

            for _ in range(self.epochs):
                loss = self.calculate_loss(self.epilon, mini_batch) # Calculate the loss
                self.optimizer.zero_grad() # Reset the gradients
                loss.backward()
                self.optimizer.step()
            
            self.model_old.load_state_dict(self.model.state_dict()) # Clone the current model to the old model