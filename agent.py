import gym
import numpy as np
import subprocess


class QAgent:

    def __init__(self, env, discount_rate=0.95, bin_size=20):
        self.env = env
        self.discount_rate = discount_rate

        # Discretize Action Space
        self.action_space = gym.spaces.Discrete(3)

        # Discretize State Space
        self.bin_storage_level = np.linspace(env.storage_level, 170, bin_size)
        self.bin_price = np.linspace(np.min(env.price_values), np.max(env.price_values), bin_size)
        self.bin_hour = np.linspace(env.hour, 24, bin_size)
        self.bin_day = np.linspace(env.day, len(env.price_values), bin_size)
        self.bins = [self.bin_storage_level, self.bin_price, self.bin_hour, self.bin_day]

        # Construct Q-table
        action_space_size = (self.action_space.n,)
        state_space_size = (bin_size,) * len(self.bins)  # TODO: do we need to do -1 ?
        self.Qtable = np.zeros(state_space_size + action_space_size)

    def discretize_state(self, state):
        discretized_state = [np.digitize(state[i], self.bins[i]) - 1 for i in range(len(state))]
        return discretized_state

    def act(self, state, epsilon=0.05, learning_rate=0.1):
        """Chooses an action based on epsilon-greedy policy, and updates the Qtable"""

        # Discretize State
        state = self.discretize_state(state)

        # Picks Action based on Epsilon Greedy
        if np.random.uniform(0, 1) < epsilon:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.Qtable[tuple(state), :])

        # Update Qtable
        next_state, reward, _ = self.env.step(action)
        next_state = self.discretize_state(next_state)
        Q_target = (reward + self.discount_rate * np.max(self.Qtable[tuple(next_state)]))
        delta = learning_rate * (Q_target - self.Qtable[tuple(state) + (action,)])
        self.Qtable[tuple(state) + (action,)] = self.Qtable[tuple(state) + (action,)] + delta

        return action


if __name__ == '__main__':
    subprocess.run(['python', 'main.py'])
