import numpy as np
import subprocess


class QAgent:

    def __init__(self, env, discount_rate=0.95, bin_size=20):
        self.env = env
        self.discount_rate = discount_rate

        # Discretize Action Space
        self.action_space = np.linspace(env.continuous_action_space.low, env.continuous_action_space.high, 3)

        # Discretize State Space
        self.bin_storage_level = np.linspace(env.storage_level, 170, bin_size)
        self.bin_price = np.linspace(np.min(env.price_values), np.max(env.price_values), bin_size)
        self.bin_hour = np.linspace(env.hour, 24, bin_size)
        self.bin_day = np.linspace(env.day, len(env.price_values), bin_size)
        self.bins = [self.bin_storage_level, self.bin_price, self.bin_hour, self.bin_day]
        # TODO: maybe a dictionary is better?

        # Construct Q-table
        action_space_size = len(self.action_space)
        state_space_size = (bin_size - 1) ** len(self.bins)  # TODO: do we need to do -1 ?
        self.Qtable = np.zeros((state_space_size, action_space_size))


    def discretize_state(self, state):
        discretized_state = [np.digitize(state[i], self.bins[i]) - 1 for i in range(len(state))]
        return discretized_state

    def act(self, state):
        # TODO: write function
        action = np.random.uniform(-1, 1)
        return action


if __name__ == '__main__':
    subprocess.run(['python', 'main.py'])
