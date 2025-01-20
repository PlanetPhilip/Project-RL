import gym
import numpy as np
from env import DataCenterEnv
import pandas as pd


class QAgent:

    def __init__(self, env, discount_rate=0.95):
        self.env = env
        self.discount_rate = discount_rate
        self.q_table_path = 'Data/q_table.npy'

        # Discretize Action Space
        self.action_space = gym.spaces.Discrete(3)

        # Discretize State Space
        features = {
            'storage_level': {'low': 0, 'high': 290, 'bin_size': 10},
            'price': {'low': np.min(env.price_values), 'high': np.max(env.price_values), 'bin_size': 10},
            'hour': {'low': 1, 'high': 25, 'bin_size': 24},
            'day': {'low': 1, 'high': len(env.price_values), 'bin_size': 10},
            'Season': {'low': 0, 'high': 3, 'bin_size': 4},
            'Avg_Price': {'low': np.min(env.test_data['Avg_Price']), 'high': np.max(env.test_data['Avg_Price']), 'bin_size': 10},
            'Rolling_Avg_Price': {'low': np.min(env.test_data['Rolling_Avg_Price']), 'high': np.max(env.test_data['Rolling_Avg_Price']), 'bin_size': 10},
            'Day_of_Week': {'low': 0, 'high': 6, 'bin_size': 7},
        }
        self.bins = [np.linspace(f['low'], f['high'], f['bin_size'] + 1) for f in features.values()]

        # Construct Q-table
        action_space_size = (self.action_space.n,)
        state_space_size = tuple(f['bin_size'] for f in features.values())
        self.Qtable = np.zeros(state_space_size + action_space_size)

    def discretize_state(self, state):
        """
        Discretize the state using bins for each feature.
        Ensure that the state and bins align correctly.
        """
        if len(state) != len(self.bins):
            raise ValueError(
                f"State length {len(state)} does not match number of bins {len(self.bins)}."
            )

        discretized_state = [
            max(0, min(np.digitize(state[i], self.bins[i], right=True) - 1, len(self.bins[i]) - 2))
            for i in range(len(state))
        ]
        return discretized_state

    def feature_engineering(self, state):

        # Function to extract seasonal information
        def get_season(date):
            if date.month in [12, 1, 2]: return 0  # Winter
            elif date.month in [3, 4, 5]: return 1  # Spring
            elif date.month in [6, 7, 8]: return 2  # Summer
            elif date.month in [9, 10, 11]: return 3  # Autumn

        # Get Season
        data = self.env.test_data
        data['PRICES'] = pd.to_datetime(data['PRICES'])
        date = data['PRICES'][self.env.day]
        print('date', date, type(date))
        Season = get_season(date)
        print('Season', Season)

        # Get Average Price

        Avg_Price = data['PRICES'].iloc[self.env.day].mean()
        #Rolling_Avg_Price = data['Avg_Price'].rolling(window=365, min_periods=1).mean()

        Day_of_Week = ...

        state += [Season, Avg_Price, Rolling_Avg_Price, Day_of_Week]

        return state

    def act(self, state, epsilon=0):
        """
        Picks action based on epsilon-greedy policy.
        """
        if np.random.uniform(0, 1) < epsilon:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.Qtable[tuple(state)])
        return action

    def update_qtable(self, state, action, next_state, reward, learning_rate=0.05):
        """
        Update the Q-table using the Q-learning update rule.
        """
        state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)

        Q_target = reward + self.discount_rate * np.max(self.Qtable[tuple(next_state)])
        delta = learning_rate * (Q_target - self.Qtable[tuple(state) + (action,)])
        self.Qtable[tuple(state) + (action,)] += delta

    def train(self, n_simulations=1000, start_epsilon=1, end_epsilon=0.05):
        """
        Train the agent using Q-learning.
        """
        epsilon = start_epsilon
        decay_rate = (end_epsilon / start_epsilon) ** (1 / (n_simulations - 1))

        for i in range(n_simulations):
            print(f'Simulation: {i + 1}')
            state = self.env.reset()
            terminated = False

            while not terminated:
                state = self.feature_engineering(state)
                # state = self.discretize_state(state)
                action = self.act(state, epsilon)
                next_state, reward, terminated = self.env.step(action)
                self.update_qtable(state, action, next_state, reward)
                state = next_state

            epsilon *= decay_rate

        # Save Q-table
        np.save(self.q_table_path, self.Qtable)

    def evaluate(self, print_transitions=False):
        """
        Evaluate the agent.
        """
        self.Qtable = np.load(self.q_table_path)
        state = self.env.reset()
        aggregate_reward = 0
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
    dataset_path = 'Data/transformed_dataset.xlsx'
    data = pd.read_excel(dataset_path)
    data['PRICES'] = pd.to_datetime(data['PRICES'])
    data.to_excel(dataset_path, index=False, engine='openpyxl')


    environment = DataCenterEnv(path_to_test_data=dataset_path)
    q_agent = QAgent(environment)

    print("Training Q-learning Agent...")
    q_agent.train(n_simulations=1000)

    print("\nEvaluating Q-learning Agent...")
    q_agent.evaluate(print_transitions=True)
