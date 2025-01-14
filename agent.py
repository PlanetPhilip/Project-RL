import gym
import numpy as np
import subprocess


class QAgent:

    def __init__(self, env, discount_rate=0.95, bin_size=10):
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
        state_space_size = (bin_size,) * len(self.bins)  # do we need to do -1 ?
        self.Qtable = np.zeros(state_space_size + action_space_size)

    def discretize_state(self, state):
        discretized_state = [np.digitize(state[i], self.bins[i]) - 1 for i in range(len(state))]
        return discretized_state

    def act(self, state, epsilon=0):

        # Picks Action based on Epsilon Greedy
        if np.random.uniform(0, 1) < epsilon:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.Qtable[tuple(state)])

        return action

    def update_qtable(self, state, action, next_state, reward, learning_rate=0.05):

        # Update State
        next_state = self.discretize_state(next_state)

        # Update Qtable
        Q_target = (reward + self.discount_rate * np.max(self.Qtable[tuple(next_state)]))
        delta = learning_rate * (Q_target - self.Qtable[tuple(state) + (action,)])
        self.Qtable[tuple(state) + (action,)] = self.Qtable[tuple(state) + (action,)] + delta

    def train(self, n_simulations=1000, start_epsilon=1, end_epsilon=0.05):

        # Initialize Epsilon
        epsilon = start_epsilon
        decay_rate = (end_epsilon / start_epsilon) ** (1 / (n_simulations - 1))

        for i in range(n_simulations):
            print(f'Simulation: {i}')

            # Initialize
            environment = self.env
            state = environment.reset()

            # Rollout
            terminated = False
            while not terminated:
                state = self.discretize_state(state)
                action = self.act(state, epsilon)
                next_state, reward, terminated = environment.step(action)
                self.update_qtable(state, action, next_state, reward)
                state = next_state
                # state = self.update(state, action)

            # Update epsilon
            epsilon = epsilon * decay_rate

        # Save Q-table
        np.save('Data/q_table.npy', self.Qtable)

    def evaluate(self, print_transitions=False):

        # Load Q-table
        self.Qtable = np.load('Data/q_table.npy')
        print(f"Explored: {100 * (1 - np.count_nonzero(self.Qtable == 0) / self.Qtable.size):.2f}%")

        # Initialize
        state = self.env.reset()
        aggregate_reward = 0

        # Rollout
        terminated = False
        while not terminated:
            state = self.discretize_state(state)
            action = self.act(state)
            next_state, reward, terminated = self.env.step(action)
            state = next_state
            aggregate_reward += reward

            if print_transitions:
                print("Action:", action)
                print("Next state:", next_state)
                print("Reward:", reward)

        print('Total reward:', aggregate_reward)


if __name__ == '__main__':
    subprocess.run(['python', 'main.py'])
