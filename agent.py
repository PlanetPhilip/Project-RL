import gym
import numpy as np
import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt


class QAgent:

    def __init__(self, agent_nr, env, discount_rate=0.95, small_reward=50000, large_reward=100000, learning_rate=0.05,
                 n_episodes=10, state_bin_size=[5, 5, 6, 7, 4], optimization_mode=False, use_rewardshaping=True):

        # Agent Details
        self.name = "QAgent"
        self.agent_nr = agent_nr
        self.env = env
        self.data = pd.read_excel('Data/train-cleaned-features.xlsx')

        # Hyperparameters
        self.discount_rate = discount_rate
        self.q_table_path = f'QTables/q_table_{agent_nr}.npy'
        self.small_reward = small_reward
        self.large_reward = large_reward
        self.learning_rate = learning_rate
        self.n_episodes = n_episodes
        self.state_bin_size = state_bin_size
        self.optimization_mode = optimization_mode
        self.use_rewardshaping = use_rewardshaping

        # Discretize Action Space
        self.action_space = gym.spaces.Discrete(3)

        # Discretize State Space
        states = {
            'storage_level': {'low': 0, 'high': 290, 'bin_size': state_bin_size[0]},
            'price': {'low': np.min(env.price_values), 'high': np.max(env.price_values), 'bin_size': state_bin_size[1]},
            'hour': {'low': 1, 'high': 25, 'bin_size': state_bin_size[2]},
            'Day_of_Week': {'low': 0, 'high': 6, 'bin_size': state_bin_size[3]},
            'Season': {'low': 0, 'high': 3, 'bin_size': state_bin_size[4]}
            }
        self.bins = [np.linspace(f['low'], f['high'], f['bin_size'] + 1) for f in states.values()]

        # Construct Q-table
        action_space_size = (self.action_space.n,)
        state_space_size = tuple(f['bin_size'] for f in states.values())
        self.Qtable = np.zeros(state_space_size + action_space_size)

        # Print
        print(f"Features: {list(states.keys())}")
        print(f"Bin sizes: {state_bin_size}")
        print("")


    def discretize_state(self, state):
        # Feature Engineering (State -> Features)
        state = self.feature_engineering(state)

        # Digitize Features
        discretized_state = [np.digitize(state[i], self.bins[i], right=True) - 1 for i in range(len(state))]
        return discretized_state

    def feature_engineering(self, state):
        """ Engineer additional features from the raw state. """

        # Get Day_of_Week from transformed dataset
        day_of_Week = self.data['Day_of_Week'].iloc[self.env.day - 1]

        # Get Season from transformed dataset
        season = self.data['Season'].iloc[self.env.day - 1]

        # Add to state
        state = np.append(state[:-1], [day_of_Week, season])
        return state

    def act(self, state, epsilon=0):
        """ Picks action based on epsilon greedy policy."""

        # Random Action
        if np.random.uniform(0, 1) < epsilon:
            action = self.action_space.sample()

        # Greedy Action
        else: action = np.argmax(self.Qtable[tuple(state)])

        return action - 1

    def potential_function(self, state, action):
        """
        Potential function for shaping the reward function.
        Add a small reward if agent buys on Friday and Saturday.
        Add a slightly larger reward if agent buys in the morning.
        Add a small reward if agent sells when it is expeisnve.
        """

        # Consider the commented out hours as ideal buying hours later,
        # And shaping the price as well on an average basis.

        # No Reward Shaping
        if not self.use_rewardshaping: return 0

        # Reward Shaping
        else:
            additional_reward = 0

            # Heuristic Information
            ideal_buying_hour = [22, 23, 0, 1, 2, 3, 4, 5, 6, 7]
            ideal_buying_day = [4, 5]
            ideal_selling_hour = [10, 11, 12, 13, 14, 19, 20]
            ideal_selling_day = [0, 2]
            state_day_of_week = state[3]
            state_hour = state[2]

            # Give large reward if agent buys on ideal buying hours
            if action == 2 and state_hour in ideal_buying_hour:
                additional_reward += self.large_reward

            # Give small reward if agent buys on ideal buying days
            if action == 2 and state_day_of_week in ideal_buying_day:
                additional_reward += self.small_reward

            # Give large reward if agent sells on ideal selling hours
            if action == 0 and state_hour in ideal_selling_hour:
                additional_reward += self.large_reward

            # Give small reward if agent sells on ideal selling days
            if action == 0 and state_day_of_week in ideal_selling_day:
                additional_reward += self.small_reward

            return additional_reward

    def update_qtable(self, state, action, next_state, reward):

        # Discretize Next State
        next_state = self.discretize_state(next_state)

        # Shape Reward   r'(s,a s') = r(s,a,s) + (gamma * P(s') - P(s))
        shaped_reward = (reward + ((self.discount_rate * self.potential_function(next_state, action))
                                   - self.potential_function(state, action)))
        # Update Qtable
        Q_target = shaped_reward + self.discount_rate * np.max(self.Qtable[tuple(next_state)])
        delta = self.learning_rate * (Q_target - self.Qtable[tuple(state) + (action,)])
        self.Qtable[tuple(state) + (action,)] = self.Qtable[tuple(state) + (action,)] + delta

    def train(self):

        # Initialize Epsilon
        epsilon = 1

        # Decay rate version 1
        decay_rate = (0.05 / 1) ** (1 / (self.n_episodes - 1))

        # If QTable exists, delete it and then make a new one, because QAgents need a completely new QTable for each training
        if self.q_table_path in os.listdir('QTables'):
            os.remove(self.q_table_path)

        for i in range(self.n_episodes):
            print(f"Episode: {i + 1}/{self.n_episodes}")

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

        # Save Q-table
        np.save(self.q_table_path, self.Qtable)
        print(f'\nQtable saved to {self.q_table_path}')

    def evaluate(self, 
                print_transitions=False,
                show_plot=False, 
                save_plot=False,
                xlim=(0,1000), 
                ylim=(-1000,1000), 
                ):

        print(self.name)
        print("Start evaluating:")
        print(f"Q-table shape: {self.Qtable.shape}")
        # print(f"Q-table: {self.Qtable}")
        print(f"Bins: {self.bins}")

        # Load Q-table
        if self.name != 'Heuristic':
            try:
                self.Qtable = np.load(self.q_table_path)
                print(f"Q-table shape: {self.Qtable.shape}")
            except Exception as e:
                print(f"Error loading Q-table: {e}")
                return

        print(f"Explored: {100 * (1 - np.count_nonzero(self.Qtable == 0) / self.Qtable.size):.2f}%")

        # Initialize
        state = self.env.reset()
        aggregate_reward = 0

        # Create a txt file to store the transition profile
        if self.use_rewardshaping:
            action_profile_filepath = f'Results/Agent_{self.agent_nr}_with_rewardshaping.txt'
        else:
            action_profile_filepath = f'Results/Agent_{self.agent_nr}_without_rewardshaping.txt'

        with open(action_profile_filepath, 'w') as f:
            f.write("Transition Profile during evaluation with reward shaping:\n")
            f.write(f"Q-table shape: {self.Qtable.shape}\n")
            # f.write(f"Q-table: {self.Qtable}\n")
            f.write(f"State bin size: {self.state_bin_size}\n")
            f.write(f"Bins length: {len(self.bins)}\n")
            f.write(f"Bins: {self.bins}\n")
            f.write(f"Discount rate: {self.discount_rate}\n")
            f.write(f"Small reward: {self.small_reward}\n")
            f.write(f"Large reward: {self.large_reward}\n")
            f.write(f"Learning rate: {self.learning_rate}\n")
            f.write(f"N episodes: {self.n_episodes}\n")

        # Rollout
        terminated = False
        while not terminated:
            with open(action_profile_filepath, 'a') as f:
                f.write(f"State: {state},")
            state = self.discretize_state(state)
            action = self.act(state)
            next_state, reward, terminated = self.env.step(action)
            state = next_state
            aggregate_reward += reward
            with open(action_profile_filepath, 'a') as f:
                f.write(f" Action: {action}, Reward: {reward}, Next state: {next_state}\n")

            if print_transitions:
                print("Action:", action)
                print("Next state:", next_state)
                print("Reward:", reward)

        print(f'Total reward: {round(aggregate_reward)}\n')


        # If agent is in optimization mode, record the hyperparameters and the total reward during each optimization run

        if self.optimization_mode:
            
            if 'Results/Agent_{self.agent_nr}_optimization_results.txt' not in os.listdir('Results'):
                file_mode = 'w'
            else:
                file_mode = 'a' 

            with open(f'Results/Agent_{self.agent_nr}_optimization_results.txt', file_mode) as f:
                f.write("Agent info during optimization:\n")
                f.write(f"Q-table shape: {self.Qtable.shape}\n")
                # f.write(f"Q-table: {self.Qtable}\n")
                f.write(f"State bin size: {self.state_bin_size}\n")
                f.write(f"Bins length: {len(self.bins)}\n")
                f.write(f"Bins: {self.bins}\n")
                f.write(f"Discount rate: {self.discount_rate}\n")
                f.write(f"Small reward: {self.small_reward}\n")
                f.write(f"Large reward: {self.large_reward}\n")
                f.write(f"Learning rate: {self.learning_rate}\n")
                f.write(f"N episodes: {self.n_episodes}\n")
                f.write(f"Total reward: {aggregate_reward}\n")
                f.write(f"\n\n")

        # Plot the action profile if show_plot is True
        if show_plot:
            print("Plotting action profile...")
            self.plot_action(action_profile_filepath, save_plot=save_plot, xlim=xlim, ylim=ylim)

        return aggregate_reward
    
    # Functions used for plotting
    def plot_action(self, 
                    action_profile_filepath, 
                    save_plot=False,
                    xlim=(0,1000), 
                    ylim=(-1000,1000)):

        # Preprocess the transition info
        transition_info = []
        with open(action_profile_filepath, "r") as infile:
            for line in infile:
                if line.startswith("State:"):
                    transition_info.append(line)

        states = []
        actions = []
        rewards = []
        # next_states = []

        for transition in transition_info:
            transition = transition.split(",")

            state = transition[0].replace('State: ', "").strip()
            action = transition[1].replace(" Action: ","").strip()
            reward = transition[2].replace(" Reward: ","").strip()
            # next_state = transition[3].replace(" Next state: ","").strip()

            # Convert string representations to lists of numbers
            state_in_num = [float(x) for x in state.strip('[]').split() if x]
            # next_state_in_num = [float(x) for x in next_state.strip('[]').split() if x]

            states.append(state_in_num)
            actions.append(float(action))
            rewards.append(float(reward))
            # next_states.append(next_state_in_num)

        # End of preprocess

        # Plot the action profile
        storage = [state[0] for state in states]
        price = [state[1] for state in states]
        hour = [state[2] for state in states]
        # day_of_week = [state[3] for state in states]
    
        if len(states[0]) == 5: # Get the season of each transition if the state set includes season
            season = [state[4] for state in states]
        
        x_axis = range(len(actions))
        y_axis1 = storage
        y_axis2 = hour
        y_axis2 = price
        y_axis3 = [action * 200 for action in actions] # Amplify the action to make it more visible
        y_axis4 = rewards
        if len(states[0]) == 5:
            y_axis5 = season

        plt.plot(x_axis, y_axis1, label='Storage', color='blue')
        plt.plot(x_axis, y_axis2, label='Hour', color='red')
        plt.plot(x_axis, y_axis3, label='Action', color='green')
        plt.plot(x_axis, y_axis4, label='Reward', color='purple')
        if len(states[0]) == 5:
            plt.plot(x_axis, y_axis5, label='Season', color='orange')

        plt.xlim(xlim)
        plt.ylim(ylim)

        plt.xlabel('Transition Step')
        plt.ylabel('State Values')
        plt.title('Action Profile of Agent ' + self.agent_nr)
        plt.legend()
        plt.grid(True)

        if save_plot:
            plt.savefig(f'Plots/Agent_{self.agent_nr}_action_profile.png')

        plt.show()



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
        return action - 1


if __name__ == '__main__':
    # Example of running QAgent with subprocess
    agent_nr = str(1)

    subprocess.run(['python', 'main.py', 
                    '--mode', 'train',
                    '--agent', 'QAgent', 
                    '--agent_nr', agent_nr, 
                    '--show_plot', 'True',
                    '--save_plot', 'True',
                    '--use_rewardshaping', 'True'
                    ])
