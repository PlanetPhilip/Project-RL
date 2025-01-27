import multiprocessing
from agent import QAgent
from env import DataCenterEnv
import torch.nn as nn
import subprocess
import os
import csv
import shutil


def extract_total_reward(output):
    # Assuming the total reward is printed in the format: "Total reward: <value>"
    explored = 0
    total_reward = 0

    for line in output.splitlines():
        if "Explored" in line:
            explored = float(line.split(":")[1].strip().replace("%", ""))
        if "Total reward:" in line:
            # Extract the numerical value and convert it to float
            total_reward = float(line.split(":")[1].strip())

    return explored, total_reward

def run_experiment(params):
    agent_nr, env_path, small_reward, large_reward, learning_rate, n_episodes, state_choice, state_bin_size, use_rewardshaping = params
    
    # Run training
    training_result = subprocess.run([
        'python', 'main.py', 
        '--mode', 'train', 
        '--agent', 'QAgent',
        '--agent_nr', str(agent_nr),
        '--path', env_path,
        '--small_reward', str(small_reward),
        '--large_reward', str(large_reward),
        '--learning_rate', str(learning_rate),
        '--n_episodes', str(n_episodes),
        '--state_choice', ",".join(state_choice),
        '--state_bin_size', ",".join(map(str, state_bin_size)),
        '--use_rewardshaping', str(use_rewardshaping)
        ])
    
    if training_result.returncode != 0:
        print(f"Error during training for Agent {agent_nr}")
        return None
    
    # Run validation
    result = subprocess.run([
        'python', 'main.py', 
        '--mode', 'validate', 
        '--agent', 'QAgent',
        '--agent_nr', str(agent_nr),
        '--path', env_path,
        '--show_plot', 'True',  # or 'False' as needed
        '--save_plot', 'True'   # or 'False' as needed
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error during validation for Agent {agent_nr}:")
        print(result.stderr)
        return None

    explored, total_reward = extract_total_reward(result.stdout)
    
    csv_file = 'Results/experiment_results_with_reward_shaping.csv'
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([agent_nr, small_reward, large_reward, learning_rate, n_episodes, explored, total_reward])
    
    return total_reward


if __name__ == '__main__':


    csv_file = 'Results/experiment_results_with_reward_shaping.csv'
    file_exists = os.path.isfile(csv_file)
    if not file_exists:
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Agent No.', 'Small Reward', 'Large Reward', 'Learning Rate', 'Episodes', 'Explored', 'Total Reward'])
    
    # Define different sets of hyperparameters
    
    # How to use it?
    # 1. fill in the following parameters in a bracket sequentially:
        # Param Set (index No. of the parameter set)
        # Small Reward
        # Large Reward
        # Learning Rate
        # Number of Episodes
    # 2. Set up the number of processes in the line below:
        # with multiprocessing.Pool(processes=10) as pool:
        # Don't make it too large otherwise each process will take too long to run

    # 3. Run the code
    # Result will be appended automatically in the csv file: Results/experiment_results_Q.csv

    # 4.If you want to plot the results, set the show_plot and save_plot parameters to True WITHIN the run_experiment function

    # Define the state name set and their bin size
    # Make sure the bin size has the same length as the state name set
    # Make sure that 'Day_of_Week' is the 4th element in the state name set, otherwise the reward shaping function 
    # Later will not work

    state_set1 = ["storage_level", "price", "hour",  'Day_of_Week']
    state_set2 = ["storage_level", "price", "hour", 'Day_of_Week', 'Season']

    # Select which data to use
    # env_path = 'Data/train.xlsx' # Raw training data
    env_path = 'Data/train_cleaned_features.xlsx' # Cleaned training data

    hyperparameter_sets = [
                          # Agents with no reward shaping
                          # Agent without season
                        #  (1, env_path, 32394, 50794, 0.01, 1000, state_set1, [10,10,24,7], False),
                        #   (2, env_path, 32394, 50794, 0.01, 1000, state_set1, [10,10,6,7], False),
                        #   (3, env_path, 32394, 50794, 0.01, 1000, state_set1, [5,10,6,7], False),
                        #   (4, env_path, 32394, 50794, 0.01, 1000, state_set1, [5,5,6,7], False),
                        #   (5, env_path, 32394, 50794, 0.01, 1000, state_set1, [10,10,6,3], False),
                        #   (6, env_path, 32394, 50794, 0.01, 1000, state_set1, [10,5,6,3], False),
                        #   (7, env_path, 32394, 50794, 0.01, 1000, state_set1, [5,10,6,7], False),
                        #   (8, env_path, 32394, 50794, 0.01, 1000, state_set1, [5,5,6,3], False),
                        #   # Agent with season
                        #   (9, env_path, 32394, 50794, 0.01, 1000, state_set2, [10,10,24,7,4], False),
                        #   (10, env_path, 32394, 50794, 0.01, 1000, state_set2, [10, 10, 6,7,4], False),
                        #   (11, env_path, 32394, 50794, 0.01, 1000, state_set2, [5,10,6,7,4], False),
                        #   (12, env_path, 32394, 50794, 0.01, 1000, state_set2, [10,5,6,7,4], False),
                        #   (13, env_path, 32394, 50794, 0.01, 1000, state_set2, [5,5,6,7,4], False),
                        #   (14, env_path, 32394, 50794, 0.01, 1000, state_set2, [5,5,6,3,4], False),
                        #   (15, env_path, 32394, 50794, 0.01, 1000, state_set2, [10,10,6,3,4], False),
                        #   (16, env_path, 32394, 50794, 0.01, 1000, state_set2, [5,5,6,3,4], False),
                          # Agents with reward shaping
                          # Agent without season
                          # Shape small
                        #   (17, env_path, 0, 100, 0.01, 1500, state_set1, [10,10,24,7], True),
                        #   (18, env_path, 0, 100, 0.01, 1500, state_set1, [5,5,6,7], True),
                        #   (19, env_path, 100, 300, 0.01, 1500, state_set1, [10,10,24,7], True),
                        #   (20, env_path, 100, 300, 0.01, 1500, state_set1, [5,5,6,7], True),
                        #   # Shape large
                        #   (21, env_path, 0, 500, 0.01, 1500, state_set1, [10,10,24,7], True),
                        #   (22, env_path, 0, 500, 0.01, 1500, state_set1, [5,5,6,7], True),
                        #   (23, env_path, 300, 1000, 0.01, 1500, state_set1, [10,10,24,7], True),
                        #   (24, env_path, 300, 1000, 0.01, 1500, state_set1, [5,5,6,7], True),
                          # Agent with season
                          # Shape small
                        #   (25, env_path, 0, 100, 0.01, 1500, state_set2, [10,10,24,7,4], True),
                        #   (26, env_path, 0, 100, 0.01, 1500, state_set2, [5,5,6,7,4], True),
                        #   (27, env_path, 100, 300, 0.01, 1500, state_set2, [10,10,24,7,4], True),
                        #   (28, env_path, 100, 300, 0.01, 1500, state_set2, [5,5,6,7,4], True),
                          # Shape large
                          (29, env_path, 0, 500, 0.01, 1500, state_set2, [10,10,24,7,4], True),
                          (30, env_path, 0, 500, 0.01, 1500, state_set2, [5,5,6,7,4], True),
                          (31, env_path, 300, 1000, 0.01, 1500, state_set2, [10,10,24,7,4], True),
                          (32, env_path, 300, 1000, 0.01, 1500, state_set2, [5,5,6,7,4], True),
                ]

    # Use multiprocessing to run experiments
    with multiprocessing.Pool(processes=4) as pool:
        pool.map(run_experiment, hyperparameter_sets)