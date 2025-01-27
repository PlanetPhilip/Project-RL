# Bayesian Optimization for QAgent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skopt
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_convergence
from agent import QAgent, TimingStats, timing_decorator
from env import DataCenterEnv

#################################################################################################################
# Customizable parameters that we can optimize or to set fixed for the QAgent
#################################################################################################################
state_set1 = ["storage_level", "price", "hour", "Day_of_Week"]
state_set2 = ["storage_level", "price", "hour", "Day_of_Week", "Season"]


# Define the fixed parameters that we do not need to optimize, but are needed for the QAgent
fixed_params = {
    "agent_nr": 33,
    "n_episodes": 10, # Number of episodes a QAgent will perform during training
    "optimization_mode": True, # Whether the QAgent is in optimization mode. ALWAYS set to True.
    "n_calls": 15, # Number of iterations of the Bayesian optimizer
    "use_rewardshaping": True, # Whether the QAgent uses reward shaping.

    # Define if we want to show and save the plots
    "show_plot": True,
    "save_plot": True
}

# Define the search space (optimizable parameters) including bin sizes
search_space = [
    Real(0, 1, name='discount_rate'),
    Real(0, 1000, name='small_reward'),
    Real(0, 20000, name='large_reward'),
    Real(0, 0.3, name='learning_rate'),
    Categorical([1], name='state_set_choice'),
    # Categorical([0, 1], name='state_set_choice'),  # 0 for state_set1, 1 for state_set2
    # Bin sizes for state_set1
    Integer(3, 10, name='storage_level_bins'),
    Integer(3, 10, name='price_bins'),
    Integer(6, 24, name='hour_bins'),
    Integer(3, 7, name='day_of_week_bins'),
    Integer(2, 4, name='season_bins')  # This will only be used for state_set2
]

# End of customizable parameters

#################################################################################################################
# Bayesian Optimizer
#################################################################################################################

def objective_function(params):
    # Unpack the parameters
    (discount_rate, small_reward, large_reward, learning_rate, 
     state_set_choice, storage_bins, price_bins, hour_bins, 
     dow_bins, season_bins) = params
    
    # Select the appropriate state set and create corresponding bin sizes
    if state_set_choice == 0:
        selected_states = state_set1
        selected_bins = [storage_bins, price_bins, hour_bins, dow_bins]
    else:
        selected_states = state_set2
        selected_bins = [storage_bins, price_bins, hour_bins, dow_bins, season_bins]
    
    # Initialize the environment
    environment = DataCenterEnv("Data/train-cleaned-features.xlsx")
    
    # Create the QAgent with the selected parameters
    agent = QAgent(
        agent_nr=f"bayesoptim_{fixed_params['agent_nr']}",
        env=environment, 
        discount_rate=discount_rate, 
        small_reward=small_reward, 
        large_reward=large_reward, 
        learning_rate=learning_rate, 
        n_episodes=fixed_params["n_episodes"],
        state_bin_size=selected_bins,
        optimization_mode=fixed_params["optimization_mode"],
        use_rewardshaping=fixed_params["use_rewardshaping"]
    )
    
    # Train and evaluate
    agent.train()
    total_reward = agent.evaluate(show_plot=fixed_params["show_plot"], save_plot=fixed_params["save_plot"])  # Added plot parameters
    
    return -total_reward


# Perform Bayesian optimization
if __name__ == "__main__":

    @timing_decorator
    def optimize(objective_function, search_space):
        result = skopt.gp_minimize(
            objective_function, 
            search_space, 
            n_calls=fixed_params["n_calls"], 
            random_state=42,
            acq_func='EI',
            verbose=True
        )
        return result
    
    # Call the optimize function with the objective_function and search_space
    result = optimize(objective_function, search_space)
    
    # Print the results
    print("Best parameters found: ", result.x)
    print("Best objective function value: ", result.fun)

    with open('Results/Agent_{self.agent_nr}_optimization_results.txt', 'a') as f:
        f.write("Final results:\n")
        f.write(f"Best parameters found: {result.x}\n")
        f.write(f"Best objective function value: {result.fun}\n")
