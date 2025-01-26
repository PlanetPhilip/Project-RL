# Bayesian Optimization for QAgent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skopt
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.plots import plot_convergence
from agent import QAgent, TimingStats, timing_decorator
from env import DataCenterEnv



# Define the fixed parameters of the QAgent that we do not want to optimize
fixed_params = {
    "agent_nr": 33,
    "n_simulations": 5,
    "state_choice": ["storage_level", "price", "hour", "Day_of_Week"],
    "state_bin_size": [10,10,24,7]
}

# Define the objective function
def objective_function(params):
    discount_rate, small_reward, large_reward, learning_rate = params
    
    # Initialize the environment with the correct path
    # environment = DataCenterEnv(TRAIN)
    environment = DataCenterEnv("Data/train.xlsx")
    
    # Create the QAgent with the environment
    agent = QAgent(f"bayesoptim_{fixed_params['agent_nr']}",
                   environment, 
                   discount_rate, 
                   small_reward, 
                   large_reward, 
                   learning_rate, 
                   fixed_params["n_simulations"], 
                   fixed_params["state_choice"], 
                   fixed_params["state_bin_size"])
    
    # Train the agent
    agent.train()

    # Evaluate the agent
    total_reward = agent.evaluate()

    # Return the negative of the total reward for minimization
    return -total_reward


# Define the search space
search_space = [
    skopt.space.Real(0, 1, name='discount_rate'),
    skopt.space.Real(0, 1000, name='small_reward'),
    skopt.space.Real(0, 20000, name='large_reward'),
    skopt.space.Real(0, 0.3, name='learning_rate'),
]



# Perform Bayesian optimization
if __name__ == "__main__":

    @timing_decorator
    def optimize(objective_function, search_space):
        result = skopt.gp_minimize(
            objective_function, 
            search_space, 
            n_calls=10, 
            random_state=42
        )
        return result
    
    # Call the optimize function with the objective_function and search_space
    result = optimize(objective_function, search_space)
    
    # Print the results
    print("Best parameters found: ", result.x)
    print("Best objective function value: ", result.fun)

    