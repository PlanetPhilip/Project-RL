import gym
import numpy as np
import subprocess
import os
import pandas as pd

class QAgent:

    def __init__(self, 
                 env, 
                 discount_rate=0.95, 
                 small_reward=50000, 
                 large_reward=100000, 
                 learning_rate=0.05, 
                 n_simulations=10,
                 state_choice=["storage_level", "price", "hour", "day"]):
        
        self.name = "QAgent"
        self.env = env
        self.discount_rate = discount_rate
        self.q_table_path = 'Data/q_table.npy'
        self.small_reward = small_reward
        self.large_reward = large_reward
        self.learning_rate = learning_rate
        self.n_simulations = n_simulations
        self.state_choice = state_choice

        # Calculate initial average price
        hour_columns = [col for col in env.test_data.columns if 'Hour' in col]
        avg_price = env.test_data[hour_columns].mean(axis=1)
        rolling_avg = avg_price.rolling(window=365, min_periods=1).mean()

        # Discretize Action Space
        self.action_space = gym.spaces.Discrete(3)

        # Discretize State Space
        states = {
            'storage_level': {'low': 0, 'high': 290, 'bin_size': 10},
            'price': {'low': np.min(env.price_values), 'high': np.max(env.price_values), 'bin_size': 10},
            'hour': {'low': 1, 'high': 25, 'bin_size': 24},
            'day': {'low': env.day, 'high': len(env.price_values), 'bin_size': 10},
            # 'Season': {'low': 0, 'high': 3, 'bin_size': 4},
            # 'Avg_Price': {'low': np.min(avg_price), 'high': np.max(avg_price), 'bin_size': 10},
            # 'Rolling_Avg_Price': {'low': np.min(rolling_avg), 'high': np.max(rolling_avg), 'bin_size': 10},
            'Day_of_Week': {'low': 0, 'high': 6, 'bin_size': 7}
        }
        
        state_space = [states[state] for state in self.state_choice]

        self.bins = [np.linspace(f['low'], f['high'], f['bin_size'] + 1) for f in states.values()]

        # Construct Q-table
        action_space_size = (self.action_space.n,)
        state_space_size = tuple(f['bin_size'] for f in states.values())
        self.Qtable = np.zeros(state_space_size + action_space_size)

    def discretize_state(self, state):
        state = self.feature_engineering(state)
        discretized_state = []
        for i in range(len(state)):
            val = np.clip(state[i], self.bins[i][0], self.bins[i][-1])
            bin_idx = np.digitize(val, self.bins[i], right=True) - 1
            bin_idx = np.clip(bin_idx, 0, len(self.bins[i])-2)
            discretized_state.append(bin_idx)
        return discretized_state

    def feature_engineering(self, state):
        """
        Engineer additional features from the raw state.
        Returns only the features defined in self.features.
        """
        data = self.env.test_data

        # Get Season - handle index bounds
        day_index = min(self.env.day, len(data['PRICES']) - 1)
        date = pd.to_datetime(data['PRICES'].iloc[day_index])
        month = date.month
        # if month in [12, 1, 2]:
        #     Season = 0  # Winter
        # elif month in [3, 4, 5]:
        #     Season = 1  # Spring
        # elif month in [6, 7, 8]:
        #     Season = 2  # Summer
        # else:
        #     Season = 3  # Autumn

        # Calculate average price across hours
        hour_columns = [col for col in data.columns if 'Hour' in col]
        Avg_Price = data[hour_columns].iloc[day_index].mean()

        # Calculate rolling average price
        # if 'Avg_Price' not in data.columns:
        #      data['Avg_Price'] = data[hour_columns].mean(axis=1)
        # Rolling_Avg_Price = data['Avg_Price'].rolling(window=365, min_periods=1).mean().iloc[day_index]

        # Get day of week
        Day_of_Week = date.dayofweek

        # Return only the features we defined bins for
        return [state[0], state[1], state[2], state[3], Day_of_Week]
    
        # return [state[0], state[1], state[2], state[3], Season, Avg_Price, Rolling_Avg_Price,
        #        Day_of_Week]

    def act(self, state, epsilon=0):
        # Picks Action based on Epsilon Greedy
        if np.random.uniform(0, 1) < epsilon:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.Qtable[tuple(state)])
        return action

    def potential_function(self, state, action):
        """
        Potential function for shaping the reward function.
        Add a small reward if agent buys on Friday and Saturday.
        Add a slightly larger reward if agent buys in the morning.
        Add a small reward if agent sells when it is expeisnve.
        """

        # Consider the commented out hours as ideal buying hours later,
        # And shaping the price as well on an average basis.

        ideal_buying_hour = [22,23,0,1,2,3,4,5,6,7]
        # ideal_buying_hour = [22,23,0,1,2,3,4,5,6,7, 8, 9]
        ideal_buying_day = [4,5]
        ideal_selling_hour = [10,11,12,13,14, 19, 20]
        # ideal_selling_hour = [11,12,13]
        ideal_selling_day = [0,2]

        state_day = state[4]
        state_hour = state[2]

        additional_reward = 0
        if action == 2 and state_hour in ideal_buying_hour:
            additional_reward += self.large_reward

        if action == 2 and state_day in ideal_buying_day:
            additional_reward += self.small_reward

        if action == 0 and state_hour in ideal_selling_hour:
            additional_reward += self.large_reward

        if action == 0 and state_day in ideal_selling_day:
            additional_reward += self.small_reward

        return additional_reward

    def update_qtable(self, state, action, next_state, reward):
        # Update State
        next_state = self.discretize_state(next_state)

        # Shape Reward   r'(s,a s') = r(s,a,s) + (gamma * P(s') - P(s))
        shaped_reward = (reward + ((self.discount_rate * self.potential_function(next_state, action)) - self.potential_function(state, action)))

        # Update Qtable  Q(s,a) = Q(s,a) + α*(R(s,a) + γ*max(Q(s',a) - Q(s,a))
        Q_target = shaped_reward + self.discount_rate * np.max(self.Qtable[tuple(next_state)])
        delta = self.learning_rate * (Q_target - self.Qtable[tuple(state) + (action,)])
        self.Qtable[tuple(state) + (action,)] = self.Qtable[tuple(state) + (action,)] + delta

    def train(self):
        # Initialize Epsilon
        epsilon = 1
        decay_rate = (0.05 / 1) ** (1 / (self.n_simulations - 1))

        for i in range(self.n_simulations):
            print(f"Simulation: {i + 1}/{self.n_simulations}")

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
    # Example of running QAgent with subprocess
    # subprocess.run(['python', 'main.py', '--mode', 'train', '--agent', 'QAgent', '--small_reward', '50000', '--large_reward', '100000', '--learning_rate', '0.05', '--n_simulations', '10'])
    subprocess.run(['python', 'main.py', '--mode', 'validate', '--agent', 'QAgent'])