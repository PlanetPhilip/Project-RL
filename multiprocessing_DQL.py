import multiprocessing
from env import DataCenterEnv
from agent2 import DQAgent, train_dqn
import torch.nn as nn
import os
import csv


def run_experiment(params):
    param_set_nr, env_path, learning_rate, hidden_layers, buffer_size, batch_size, min_replay_size, n_episodes, activation_functions = params
    return train_dqn(param_set_nr, env_path, learning_rate, hidden_layers, buffer_size, batch_size, min_replay_size, n_episodes, activation_functions)

if __name__ == '__main__':

    result_csv = 'Results/experiment_results_DQL.csv'
    file_exists = os.path.isfile(result_csv)
    if not file_exists:
        with open(result_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Param Set', 'Small Reward', 'Large Reward', 'Learning Rate', 'Episodes', 'Explored', 'Total Reward'])

    # Define different sets of hyperparameters
    hyperparameter_sets = [
        (1, 'Data/train.xlsx', 1e-3, [128, 64, 32], 50000, 32, 1000, 10000, [nn.Tanh(), nn.Tanh(), nn.Tanh()]),
        (2, 'Data/train.xlsx', 1e-3, [128, 64, 32], 50000, 32, 1000, 10000, [nn.Softmax(), nn.Softmax(), nn.Softmax()]),
        (3, 'Data/train.xlsx', 1e-3, [256, 128, 64], 100000, 64, 2000, 10000, [nn.Tanh(), nn.Tanh(), nn.Tanh()]),
        (4, 'Data/train.xlsx', 1e-3, [256, 128, 64], 100000, 64, 2000, 10000, [nn.Softmax(), nn.Softmax(), nn.Softmax()]),
        (5, 'Data/train.xlsx', 1e-3, [128, 64, 32], 50000, 32, 1000, 10000, [nn.Tanh(), nn.Tanh(), nn.Tanh()]),
        (6, 'Data/train.xlsx', 5e-4, [128, 64, 32], 50000, 32, 1000, 10000, [nn.Tanh(), nn.Tanh(), nn.Tanh()]),
        (7, 'Data/train.xlsx', 5e-4, [256, 128, 64], 100000, 64, 2000, 10000, [nn.Tanh(), nn.Tanh(), nn.Tanh()]),
        (8, 'Data/train.xlsx', 5e-4, [256, 128, 64], 100000, 64, 2000, 10000, [nn.Softmax(), nn.Softmax(), nn.Softmax()]),

    ]

    # Use multiprocessing to run experiments
    with multiprocessing.Pool(processes=8) as pool:
        results = pool.map(run_experiment, hyperparameter_sets)

    # Process results
    for result in results:
        print(result)