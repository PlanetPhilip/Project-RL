import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from mpl_toolkits import mplot3d
from matplotlib import cm
import pandas as pd
import seaborn as sns

# Set the seed for reproducibility
seed = 7
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


class DQN:
    def __init__(self, learning_rate):
        # Features / Action Space
        input_features = ...
        action_space = ...

        # DNN
        self.dense1 = nn.Linear(in_features=input_features, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=64)
        self.dense3 = nn.Linear(in_features=64, out_features=32)
        self.dense4 = nn.Linear(in_features=32, out_features=action_space)

        # Optimizer (use any optimizer e.g. RMSprob)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        # Forward Pass (use any activation function)
        x = torch.tanh(self.dense1(x))
        x = torch.tanh(self.dense2(x))
        x = torch.tanh(self.dense3(x))
        x = self.dense4(x)
        return x

    def learn(self):
        # Sample random batch of transitions
        ...

        # Loss Function
        ...

        # Gradient Descent
        ...


class ExperienceReplay:
    def __init__(self, buffer_size):

        self.replay_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = ...

        self.min_replay_size = ...
        # Initialize Experience Replay Buffer randomly
        for _ in range(self.min_replay_size):
            ...

    def add_data(self, data):
        self.replay_buffer.append(data)

    def sample(self):
        ...

    def add_reward(self, reward):
        self.reward_buffer.append(reward)


class DQAgent:
    def __init__(self, env):
        self.env = env
        self.action_space = gym.spaces.Discrete(3)

        # Initialize replay buffer
        self.replay_memory = ExperienceReplay()

        # Constructs DNN
        self.online_network = DQN()

    def act(self, o):
        # Define epsilon (decay)
        epsilon = ...

        # Picks Action based on Epsilon Greedy
        if np.random.uniform(0, 1) < epsilon:
            action = self.action_space.sample()
        else:
            action = ...

        return action

    def train(self, n_simulations=1000):

        # Initialize
        o = self.env.reset(seed=seed)
        total_reward = 0

        for i in range(n_simulations):

            # Rollout
            action = self.act(o)  # Choose action
            new_o, r, terminated, truncated = self.env.step()   # Next state
            self.replay_memory.add_reward((o, r, done, new_o))  # Add transition to buffer
            done = terminated or truncated
            o = new_o
            total_reward += r  # Update total reward

            # Finished
            if done:
                self.replay_memory.add_reward(r)  # Add reward to buffer

            # Learn
            self.online_network.learn()

