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
import os

# Set the seed for reproducibility
seed = 7
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


class DQN(nn.Module):
    def __init__(self, env, learning_rate=5e-4, hidden_layers=[128, 64, 32], activation_functions=[nn.Tanh(), nn.Tanh(), nn.Tanh()]):
        super(DQN, self).__init__()

        # Features / Action Space
        input_features = len(env.observation())
        action_space = gym.spaces.Discrete(3).n

        # DNN with customizable hidden layers
        layers = []
        in_features = input_features
        for out_features, activation_function in zip(hidden_layers, activation_functions):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(activation_function)
            in_features = out_features

        layers.append(nn.Linear(in_features, action_space))
        self.model = nn.Sequential(*layers)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.model(x)
    

class ExperienceReplay:
    def __init__(self, env, buffer_size=50000, batch_size=32, min_replay_size=1000):
        self.env = env
        self.action_space = gym.spaces.Discrete(3)
        self.batch_size = batch_size

        # Set Replay Buffer shape
        self.replay_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque([-200.0], maxlen=100)
        self.min_replay_size = min_replay_size

        # Initialize Replay Buffer with random transitions
        o = env.reset()                                             # Reset env, get obs
        for _ in range(self.min_replay_size):
            a = self.action_space.sample()                          # Choose action (random)
            new_o, r, t1 = env.step(a)                              # Get Next state
            self.replay_buffer.append((o, a, r, t1, new_o))         # Add transition to buffer
            o = new_o                                               # Update state
            if t1: o = env.reset()                                  # Finished -> reset env

    def add_data(self, data):
        self.replay_buffer.append(data)

    def add_reward(self, reward):
        self.reward_buffer.append(reward)

    def sample(self, batch_size):

        # Sample n transitions from replay_buffer
        transitions = random.sample(self.replay_buffer, batch_size)
        observations, actions, rewards, terminations, new_observations = map(np.asarray, zip(*transitions))

        # Reshape to tensors (for pytorch)
        observations_t = torch.as_tensor(observations, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        terminations_t = torch.as_tensor(terminations, dtype=torch.float32).unsqueeze(-1)
        new_observations_t = torch.as_tensor(new_observations, dtype=torch.float32)

        return observations_t, actions_t, rewards_t, terminations_t, new_observations_t


class DQAgent:
    def __init__(self, env, learning_rate=5e-4, hidden_layers=[128, 64, 32], buffer_size=50000, batch_size=32, min_replay_size=1000):
        self.env = env
        self.action_space = gym.spaces.Discrete(3)

        # Initialize replay buffer
        self.replay_memory = ExperienceReplay(env, buffer_size, batch_size, min_replay_size)

        # Construct DNN
        self.online_network = DQN(env, learning_rate, hidden_layers)

        # Target Network
        self.target_network = DQN(env, learning_rate, hidden_layers)
        self.target_network.load_state_dict(self.online_network.state_dict())

    def act(self, step, o, start_epsilon=1, end_epsilon=0.05, epsilon_decay=10000):
        # Define epsilon (decay)
        epsilon = np.interp(step, [0, epsilon_decay], [start_epsilon, end_epsilon])

        # Picks Action based on Epsilon Greedy
        if np.random.uniform(0, 1) < epsilon:
            action = self.action_space.sample()
        else:
            obs_t = torch.as_tensor(o, dtype=torch.float32)
            q_values = self.online_network(obs_t.unsqueeze(0))
            max_q_index = torch.argmax(q_values, dim=1)[0]
            action = max_q_index.detach().item()

        return action, epsilon

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def train(self, n_simulations=250000):

        # Initialize
        o = self.env.reset()
        average_reward_list = [-200]
        total_reward = 0

        for i in range(n_simulations):

            # Rollout
            a, e = self.act(i, o)                                      # Choose action
            new_o, r, t1 = self.env.step(a)                            # Get Next state
            self.replay_memory.add_data((o, a, r, t1, new_o))          # Add transition to buffer
            o = new_o                                                  # Update state
            total_reward += r                                          # Update total reward

            # Finished
            if t1:
                o = self.env.reset()                                # Reset env, get obs
                self.replay_memory.add_reward(total_reward)         # Add total reward to buffer
                total_reward = 0                                    # Reset total reward

            # Learn
            self.learn()

            # Update Average Reward List (every 100 episodes)
            if (i+1) % 100 == 0: average_reward_list.append(np.mean(self.replay_memory.reward_buffer))

            # Update Target Network
            target_update_frequency = 250
            if i % target_update_frequency == 0:
                agent.update_target_network()

            # Print Results
            if (i + 1) % 10000 == 0:
                print(20 * '--')
                print('Step', i)
                print('Epsilon', e)
                print('Avg Rew', np.mean(self.replay_memory.reward_buffer))
                print()

        return average_reward_list

    def learn(self, discount_rate=0.99):

        # Sample random batch of transitions
        observations_t, actions_t, rewards_t, dones_t, new_observations_t \
            = self.replay_memory.sample(self.replay_memory.batch_size)

        # Target Values
        target_q_values = self.online_network(new_observations_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = rewards_t + discount_rate * (1 - dones_t) * max_target_q_values

        # Loss (MSE)
        q_values = self.online_network(new_observations_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
        loss = F.mse_loss(action_q_values, targets.detach())

        # Gradient Descent
        self.online_network.optimizer.zero_grad()
        loss.backward()
        self.online_network.optimizer.step()

def train_dqn(param_set_nr, env_path, learning_rate, hidden_layers, buffer_size, batch_size, min_replay_size, n_simulations):
    env = DataCenterEnv(env_path)
    agent = DQAgent(env, learning_rate, hidden_layers, buffer_size, batch_size, min_replay_size)
    avg_rewards = agent.train(n_simulations)

    # Plot the behaviour of average reward
    plt.plot(1000*(np.arange(len(avg_rewards))+1), avg_rewards)
    plt.axhline(y=-110, color='r', linestyle='-')
    plt.title(f'Average reward over the past 100 simulations\n Simulation: {param_set_nr}')
    plt.xlabel('Number of simulations')
    plt.legend(['Double DQN', 'Vanilla DQN', 'Benchmark solving the game'])
    plt.ylabel('Average reward')
    plt.show()

    return avg_rewards

from env import DataCenterEnv
environment = DataCenterEnv('Data/train.xlsx')
agent = DQAgent(environment)
avg_rewards = agent.train()




