import gym
import numpy as np
import subprocess
import os
import pandas as pd
import time
import functools
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class TimingStats:
    def __init__(self):
        self.function_times = defaultdict(list)
        self.agent_nr = None  # Will be set when instantiated by an agent
        
    def add_timing(self, func_name, execution_time):
        self.function_times[func_name].append(execution_time)
        
    def get_stats(self):
        stats = {}
        for func_name, times in self.function_times.items():
            stats[func_name] = {
                'total_calls': len(times),
                'total_time': sum(times),
                'average_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        return stats
    
    def print_stats(self):
        stats = self.get_stats()
        agent_str = f" for Agent {self.agent_nr}" if self.agent_nr is not None else ""
        print(f"\nFunction Timing Statistics{agent_str}:")
        print("-" * 80)
        print(f"{'Function Name':<30} {'Calls':<10} {'Total(s)':<12} {'Avg(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12}")
        print("-" * 80)
        for func_name, data in stats.items():
            print(f"{func_name:<30} {data['total_calls']:<10} {data['total_time']:<12.3f} "
                  f"{data['average_time']*1000:<12.3f} {data['min_time']*1000:<12.3f} "
                  f"{data['max_time']*1000:<12.3f}")

def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Get the instance (self) from args and update its timing stats
        if args and hasattr(args[0], 'timing_stats'):
            args[0].timing_stats.add_timing(func.__name__, end_time - start_time)
        
        return result
    return wrapper

class QAgent:

    def __init__(self, 
                 agent_nr,
                 env, 
                 discount_rate=0.95, 
                 small_reward=50000, 
                 large_reward=100000, 
                 learning_rate=0.05, 
                 n_simulations=10,
                 state_choice=["storage_level", "price", "hour", "Day_of_Week"],
                 state_bin_size=[5,5,6,7],
                 optimization_mode = False,
                 use_rewardshaping = True
                 ):
        
        self.name = "QAgent"
        self.agent_nr = agent_nr
        self.env = env
        self.discount_rate = discount_rate
        self.q_table_path = f'QTables/q_table_{agent_nr}.npy'
        self.small_reward = small_reward
        self.large_reward = large_reward
        self.learning_rate = learning_rate
        self.n_simulations = n_simulations
        self.state_choice = state_choice
        self.state_bin_size = state_bin_size
        self.optimization_mode = optimization_mode
        self.use_rewardshaping = use_rewardshaping

        # Load transformed dataset
        self.transformed_data = pd.read_excel('Data/train-cleaned-features.xlsx')


        # Add timing stats instance
        self.timing_stats = TimingStats()
        self.timing_stats.agent_nr = agent_nr  # Set the agent number

        # Discretize Action Space
        self.action_space = gym.spaces.Discrete(3)

        # Define the bin size for each state. If not defined, set 1 as default.
        custom_bin_size = {
            "storage_level": 1,
            "price": 1,
            "hour": 1,
            "Season": 1,
            "Day_of_Week": 1
        }

        for state, bin_size in zip(self.state_choice, self.state_bin_size):
            custom_bin_size[state] = int(bin_size)

        # Discretize State Space
        states = {
            'storage_level': {'low': 0, 'high': 290, 'bin_size': custom_bin_size['storage_level']},
            'price': {'low': np.min(env.price_values), 'high': np.max(env.price_values), 'bin_size': custom_bin_size['price']},
            'hour': {'low': 1, 'high': 25, 'bin_size': custom_bin_size['hour']},
            # 'day': {'low': env.day, 'high': len(env.price_values), 'bin_size': custom_bin_size['day']},
            'Season': {'low': 0, 'high': 3, 'bin_size': custom_bin_size['Season']},
            'Day_of_Week': {'low': 0, 'high': 6, 'bin_size': custom_bin_size['Day_of_Week']}
        }
        
        # Create state_space for all possible states
        self.all_state_bins = {state: np.linspace(states[state]['low'], states[state]['high'], states[state]['bin_size'] + 1) for state in states}

        # Create state_space based on the chosen states
        state_space = {state: states[state] for state in self.state_choice if state in states}

        self.bins = [np.linspace(f['low'], f['high'], f['bin_size'] + 1) for f in state_space.values()]

        # Print the lens of the bins
        print(f"Bins: {self.bins}")
        print(f"Bins length: {len(self.bins)}")
        print(f"State space: {state_space}")

        # Construct Q-table
        action_space_size = (self.action_space.n,)
        state_space_size = tuple(f['bin_size'] for f in state_space.values())
        self.Qtable = np.zeros(state_space_size + action_space_size)

    @timing_decorator
    def discretize_state(self, state):
        # # Debugging: Print the state before discretization
        # print(f"State before discretization: {state}")

        state = self.feature_engineering(state)
        state = np.array(state)  # Convert to numpy array once
        
        discretized_state = []  # Initialize as an empty list
        
        for i in range(len(state)):
            # Ensure the value is within the bin range
            val = np.clip(state[i], self.bins[i][0], self.bins[i][-1])
            bin_idx = np.digitize(val, self.bins[i], right=True) - 1
            # Ensure the index is within bounds
            discretized_idx = np.clip(bin_idx, 0, len(self.bins[i]) - 2)
            discretized_state.append(discretized_idx)  # Append the value instead of assigning by index
        
        return discretized_state

    @timing_decorator
    def feature_engineering(self, state):
        """
        Engineer additional features from the raw state.
        Returns only the features defined in self.state_choice.
        """

        # Get Day_of_Week from transformed dataset
        day_of_Week = self.transformed_data['Day_of_Week'].iloc[self.env.day - 1]

        # Get Season from transformed dataset
        season =  self.transformed_data['Season'].iloc[self.env.day - 1]
        print('season', season)

        # Add to state
        state = state[:-1] + day_of_Week + season
        return state

    @timing_decorator
    def act(self, state, epsilon=0):
        # Picks Action based on Epsilon Greedy
        if np.random.uniform(0, 1) < epsilon:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.Qtable[tuple(state)])
        return action

    @timing_decorator
    def potential_function(self, state, action):
        """
        Potential function for shaping the reward function.
        Add a small reward if agent buys on Friday and Saturday.
        Add a slightly larger reward if agent buys in the morning.
        Add a small reward if agent sells when it is expeisnve.
        """

        # Consider the commented out hours as ideal buying hours later,
        # And shaping the price as well on an average basis.

        if not self.use_rewardshaping:
            return 0

        else:

            ideal_buying_hour = [22,23,0,1,2,3,4,5,6,7]
            # ideal_buying_hour = [22,23,0,1,2,3,4,5,6,7, 8, 9]

            ideal_buying_day = [4,5]
            ideal_selling_hour = [10,11,12,13,14, 19, 20]

            # ideal_selling_hour = [11,12,13]
            ideal_selling_day = [0,2]

            state_day_of_week = state[3]
            state_hour = state[2]

            additional_reward = 0

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

    @timing_decorator
    def update_qtable(self, state, action, next_state, reward):
        # # Debugging: Print the next state before accessing
        # print(f"Next State in update_qtable: {next_state}")

        # Ensure next_state is discretized
        next_state = self.discretize_state(next_state)

        # # Debugging: Print the discretized next state
        # print(f"Discretized Next State: {next_state}")

        # Shape Reward   r'(s,a s') = r(s,a,s) + (gamma * P(s') - P(s))
        shaped_reward = (reward + (
                    (self.discount_rate * self.potential_function(next_state, action)) - self.potential_function(state,
                                                                                                                 action)))

        # Update Qtable
        Q_target = shaped_reward + self.discount_rate * np.max(self.Qtable[tuple(next_state)])
        delta = self.learning_rate * (Q_target - self.Qtable[tuple(state) + (action,)])
        self.Qtable[tuple(state) + (action,)] = self.Qtable[tuple(state) + (action,)] + delta

    @timing_decorator
    def train(self):
        start_time = time.time()
        # Initialize Epsilon
        epsilon = 1

        # Decay rate version 1
        decay_rate = (0.05 / 1) ** (1 / (self.n_simulations - 1))

        # # Decay rate version 2
        # decay_rate = ((0.05 / 1) ** (1 / (self.n_simulations - 1))) / 2

        # If QTable exists, delete it and then make a new one, because QAgents need a completely new QTable for each training
        if self.q_table_path in os.listdir('QTables'):
            os.remove(self.q_table_path)

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

            # # Save states in a txt file to observe what information is stored in the state
            # if 'Data/states.txt' not in os.listdir("Data"):
            #     mode = 'w'
            # else:
            #     mode = 'a'
            # with open('Data/states.txt', mode) as f:
            #     f.write(str(state) + '\n')

        # Save Q-table
        np.save(self.q_table_path, self.Qtable)
        print(f'\nQtable saved to {self.q_table_path}')

        # Print timing statistics at the end of training
        training_time = time.time() - start_time
        print(f"\nTotal training time: {training_time:.2f} seconds")
        self.timing_stats.print_stats()

    @timing_decorator
    def evaluate(self, 
                print_transitions=False,
                show_plot=False, 
                save_plot=False,
                xlim=(0,1000), 
                ylim=(-1000,1000), 
                ):
        
        start_time = time.time()
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
            f.write(f"State choice: {self.state_choice}\n")
            f.write(f"State bin size: {self.state_bin_size}\n")
            f.write(f"Bins length: {len(self.bins)}\n")
            f.write(f"Bins: {self.bins}\n")
            f.write(f"Discount rate: {self.discount_rate}\n")
            f.write(f"Small reward: {self.small_reward}\n")
            f.write(f"Large reward: {self.large_reward}\n")
            f.write(f"Learning rate: {self.learning_rate}\n")
            f.write(f"N simulations: {self.n_simulations}\n")

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

        # Print timing statistics at the end of evaluation
        evaluation_time = time.time() - start_time
        print(f"\nTotal evaluation time: {evaluation_time:.2f} seconds")
        self.timing_stats.print_stats()

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
                f.write(f"State choice: {self.state_choice}\n")
                f.write(f"State bin size: {self.state_bin_size}\n")
                f.write(f"Bins length: {len(self.bins)}\n")
                f.write(f"Bins: {self.bins}\n")
                f.write(f"Discount rate: {self.discount_rate}\n")
                f.write(f"Small reward: {self.small_reward}\n")
                f.write(f"Large reward: {self.large_reward}\n")
                f.write(f"Learning rate: {self.learning_rate}\n")
                f.write(f"N simulations: {self.n_simulations}\n")
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

        plt.plot(x_axis, y_axis1, label='Storage', color='blue')
        plt.plot(x_axis, y_axis2, label='Hour', color='red')
        plt.plot(x_axis, y_axis3, label='Action', color='green')
        plt.plot(x_axis, y_axis4, label='Reward', color='purple')

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
        return action


if __name__ == '__main__':
    # Example of running QAgent with subprocess
    agent_nr = str(2)

    # subprocess.run(['python', 'main.py', 
    #                 '--mode', 'train', 
    #                 '--agent', 'QAgent', 
    #                 '--agent_nr', agent_nr,
    #                 '--small_reward', '0', 
    #                 '--large_reward', '30000', 
    #                 '--learning_rate', '0.01', 
    #                 '--n_simulations', '3', 
    #                 '--state_choice', ",".join(["storage_level", "price", "hour", "Day_of_Week", "Season"]),
    #                 '--state_bin_size', ",".join(map(str, [10, 10, 24, 7, 4]))
    #                 ])
    
    subprocess.run(['python', 'main.py', 
                    '--mode', 'validate', 
                    '--agent', 'QAgent', 
                    '--agent_nr', agent_nr, 
                    '--show_plot', 'True', 
                    '--save_plot', 'True',
                    '--use_rewardshaping', 'True'
                    ])

