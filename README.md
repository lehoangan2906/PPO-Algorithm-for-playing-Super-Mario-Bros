# Introduction
This project aims to train an agent to play Super Mario Bros using reinforcement learning (RL). The goal is to enable the agent to navigate through the game's first level (World 1-1) by learning optimal actions through trial and error. The challenge lies in processing high-dimensional visual inputs (game frames) and making real-time decisions in a dynamic environment, which makes RL a suitable approach for this task.

---

#Project Pipeline
The project follows a structured pipeline to preprocess the environment, train the agent, and evaluate its performance:

- **Environment Setup**: The Super Mario Bros environment is set up using the `gym-super-mario-bros` and `nes-py` libraries. The action space is simplified to two actions: "right" and "right + jump" to reduce complexity.
- **Preprocessing**: Game frames are preprocessed to make them suitable for the RL agent:
  - Frame skipping (every 4th frame) to reduce computational load.
  - Conversion to grayscale to simplify input data.
  - Resizing to 84x84 pixels for efficient processing.
  - Stacking 4 consecutive frames to capture temporal dynamics.


- **Agent Training**: A custom `Proximal Policy Optimization` (PPO) agent is implemented from scratch to interact with the environment, collect experiences, and optimize its policy.
- **Model Optimization**: The agent uses a custom `convolutional neural network` (CNN) model (adjusted kernel sizes and strides) to process observations and make decisions. Training involves mini-batch optimization with the Adam optimizer.
- **Checkpointing and Logging**: The agent's progress is saved periodically as checkpoints, and training metrics (episode and average rewards) are logged to a CSV file for analysis.
- **Evaluation**: The agent's performance is evaluated by rendering the game (optional) and monitoring average rewards over episodes.

---

#About the Chosen Model
The model architecture is a custom-designed actor-critic network, implemented from scratch using PyTorch:

- **Actor Network**: A `CNN` that processes the preprocessed game frames (4 stacked 84x84 grayscale images) and outputs action probabilities for the simplified action space.
- **Critic Network**: A similar CNN that estimates the value function to guide the agent's learning by predicting future rewards.
- **Algorithm**: The `Proximal Policy Optimization` (PPO) algorithm is used for training, balancing exploration and exploitation with clipped objective functions and Generalized Advantage Estimation (GAE) for stable learning.
- Why PPO?: PPO was chosen for its simplicity, stability, and effectiveness in handling continuous and discrete action spaces, making it suitable for the dynamic Super Mario Bros environment.

---

#Installation
To run this project, install the required dependencies listed in requirements.txt:
```bash
pip install -r requirements.txt
```

Ensure you have Python 3.8+ installed. The project uses PyTorch for model implementation and Gym for the environment.

---

#Usage

1. Clone the repository: 
```bash
git clone https://bit.ly/Super-Mario-RL
```

2. Navigate to the project directory:

```bash
cd Super-Mario-RL
```

3. Run the main script to start training:
```bash
python main.py
```
= Set `train = True` in main.py to train the agent, or `train = False` to run the current policy.
- Use `show_game = True` to render the game during training (computationally expensive).

---

#Results
Training progress is logged in `model/training_data.csv`, and model checkpoints are saved in `model/checkpoints/`. The agent's performance can be evaluated by monitoring the average reward over episodes, with higher rewards indicating better navigation through the level.

---

#Future Improvements
- Experiment with different hyperparameters (e.g., learning rates, batch sizes) to improve performance.
- Expand the action space to include more actions (e.g., "left", "jump") for greater flexibility.
- Explore other RL algorithms like DQN or A3C for comparison.
- Enhance the model architecture with deeper networks or attention mechanisms for better feature extraction.

---

#Acknowledgments
This project builds on the gym-super-mario-bros library for the game environment and uses PyTorch for model implementation. Inspiration was drawn from standard RL practices and PPO implementations in academic literature.
