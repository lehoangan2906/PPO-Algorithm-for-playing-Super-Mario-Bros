import csv
import torch
import os

def save_checkpoint(policy_old, episode, save_directory):
    """
    Save model checkpoint.

    Parameters:
    - policy_old: Policy model to be saved.
    - episode: Episode number for naming the checkpoint file.
    - save_directory: Directory to save the checkpoint file.
    """
    filename = os.path.join(save_directory + "/checkpoints", f'checkpoint_{episode}.pth')
    torch.save(policy_old.state_dict(), filename)
    print(f'Checkpoint saved to \'{filename}\'')

def load_checkpoint(saves_directory, agent, start):
    """
    Load a model checkpoint if available.

    Parameters:
    - saves_directory: Directory containing checkpoints.
    - agent: Agent object with models to load checkpoints into.
    - start: Episode number to start from.

    Returns:
    - agent: Updated agent object.
    """
    checkpoint_dir = os.path.join(saves_directory, "checkpoints")
    if os.path.exists(checkpoint_dir):
        saved_files = os.listdir(checkpoint_dir)
        checkpoint_files = [filename for filename in saved_files if filename.startswith("checkpoint_") and filename.endswith(".pth")]
        if checkpoint_files:
            if start == 0:
                latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
                episode_number = int(latest_checkpoint.split('_')[1].split('.')[0])
                agent.episode = episode_number
                agent.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, latest_checkpoint)))
                agent.model_old.load_state_dict(torch.load(os.path.join(checkpoint_dir, latest_checkpoint)))
            else:
                agent.episode = start
                checkpoint = f"checkpoint_{start}.pth"
                if checkpoint in checkpoint_files:
                    agent.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint)))
                    agent.model_old.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint)))
                else:
                    print(f"Checkpoint {checkpoint} not found. Starting from episode {start}.")

            print(f'Resuming training from checkpoint \'{agent.episode}\'.')
        else:
            print("No checkpoint files found.")

    return agent

def write_to_csv(save_dir, filename, episode, reward):
    """
    Write episode and reward to a CSV file.

    Parameters:
    - save_dir: Directory to save the CSV file.
    - filename: Name of the CSV file.
    - episode: Episode number.
    - reward: Total reward for the episode.
    """
    with open(os.path.join(save_dir, filename), mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode, reward])
