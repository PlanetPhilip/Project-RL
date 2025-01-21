import gym
import numpy as np
import subprocess


class QAgent:

    def __init__(self, env, discount_rate=0.95):
        self.name = 'QAgent'
        self.env = env
        self.discount_rate = discount_rate
        self.q_table_path = 'Data/q_table.npy'

        # Discretize Action Space
        self.action_space = gym.spaces.Discrete(3)

        # Discretize State Space
        states = {
            'storage_level': {'low': 0, 'high': 290, 'bin_size': 10},
            'price': {'low': np.min(env.price_values), 'high': np.max(env.price_values), 'bin_size': 10},
            'hour': {'low': 1, 'high': 25, 'bin_size': 24},
            'day': {'low': env.day, 'high': len(env.price_values), 'bin_size': 10}
        }
        self.bins = [np.linspace(f['low'], f['high'], f['bin_size'] + 1) for f in states.values()]

        # Construct Q-table
        action_space_size = (self.action_space.n,)
        state_space_size = tuple(f['bin_size'] for f in states.values())
        self.Qtable = np.zeros(state_space_size + action_space_size)

    def discretize_state(self, state):
        # print(np.digitize(state[0], self.bins[0], right=True), state[0], self.bins[0])
        discretized_state = [np.digitize(state[i], self.bins[i], right=True) - 1 for i in range(len(state))]
        return discretized_state

    def act(self, state, epsilon=0):

        # Picks Action based on Epsilon Greedy
        if np.random.uniform(0, 1) < epsilon:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.Qtable[tuple(state)])

        return action

    def potential_function(self, state):
        return 0

    def update_qtable(self, state, action, next_state, reward, learning_rate=0.05):

        # Update State
        next_state = self.discretize_state(next_state)

        # Shape Reward   R(s,a) = R(s,a) - P(s) + y*P(s')
        shaped_reward = (reward - self.potential_function(state)
                         + self.discount_rate * self.potential_function(next_state))

        # Update Qtable  Q(s,a) = Q(s,a) + α*(R(s,a) + γ*max(Q(s',a) - Q(s,a))
        Q_target = shaped_reward + self.discount_rate * np.max(self.Qtable[tuple(next_state)])
        delta = learning_rate * (Q_target - self.Qtable[tuple(state) + (action,)])
        self.Qtable[tuple(state) + (action,)] = self.Qtable[tuple(state) + (action,)] + delta

    def train(self, n_simulations=1000, start_epsilon=1, end_epsilon=0.05):

        # Initialize Epsilon
        epsilon = start_epsilon
        decay_rate = (end_epsilon / start_epsilon) ** (1 / (n_simulations - 1))

        for i in range(n_simulations):
            print(f"Simulation: {i + 1}/{n_simulations}")

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

            # Update epsilon
            epsilon = epsilon * decay_rate

            # Save states in a txt file to observe what information is stored in the state
            if 'Data/states.txt' not in os.listdir("Data"):
                mode = 'w'
            else:
                mode = 'a'
            with open('Data/states.txt', mode) as f:
                f.write(str(state) + '\n')

        # Save Q-table
        np.save(self.q_table_path, self.Qtable)
        print(f'\nQtable saved to {self.q_table_path}')

    def evaluate(self, print_transitions=False):
        print(self.name)

        # Load Q-table
        if self.name != 'Heuristic': self.Qtable = np.load(self.q_table_path)
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

        print(f'Total reward: {round(aggregate_reward)}\n')


class Heuristic(QAgent):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'Heuristic'

    def train(self, a=0, b=0, c=0):
        print("You don't need to train a heuristic!")

    def act(self, state, epsilon=0):

        # Heuristic: Picks Action based on time of day
        if state[2] in [1, 2, 3, 4, 5, 6, 7, 8, 17, 22, 23, 24]:
            action = 2
        else:
            action = 1

        return action


if __name__ == '__main__':
    subprocess.run(['python', 'main.py','--mode', 'validate', '--agent', 'QAgent'])






