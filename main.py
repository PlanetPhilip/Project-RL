from env import DataCenterEnv
from agent import QAgent, Heuristic
from Utils.args import parse_train_arguments, parse_evaluate_arguments
import sys
import os 

TRAIN = 'Data/train_cleaned_features.xlsx'
VALIDATE = 'Data/validate.xlsx'

def train(agent, agent_nr, path, small_reward, large_reward, learning_rate, n_simulations, state_choice, state_bin_size, use_rewardshaping):
    environment = DataCenterEnv(path or TRAIN)
    agent = globals()[agent](
        env=environment,
        agent_nr=agent_nr,
        small_reward=small_reward, 
        large_reward=large_reward, 
        learning_rate=learning_rate, 
        n_simulations=n_simulations,
        state_choice=state_choice,
        state_bin_size=state_bin_size,
        use_rewardshaping=use_rewardshaping
    )


    print(f"Training agent {agent_nr}...")
    agent.train()

def validate(agent, agent_nr, path, show_plot=False, save_plot=False, use_rewardshaping=True):
    try:
        environment = DataCenterEnv(path or VALIDATE)
        agent = globals()[agent](
            env=environment,
            agent_nr=agent_nr,
            use_rewardshaping=use_rewardshaping
        )
        agent.evaluate(show_plot=show_plot, save_plot=save_plot)
        
    except Exception as e:
        print(f"\nError during validation for Agent {agent_nr}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Command Line Support
    if len(sys.argv) > 1:
        if '--mode' not in sys.argv or sys.argv[sys.argv.index('--mode') + 1] == 'train':
            mode, agent, agent_nr, path, small_reward, large_reward, learning_rate, n_simulations, state_choice, state_bin_size, use_rewardshaping = parse_train_arguments()
            locals()[mode](agent, agent_nr, path, small_reward, large_reward, learning_rate, n_simulations, state_choice, state_bin_size, use_rewardshaping)
        else:
            mode, agent, agent_nr, path, show_plot, save_plot, use_rewardshaping = parse_evaluate_arguments()
            locals()[mode](agent, agent_nr, path, show_plot, save_plot, use_rewardshaping)
    else:
        # Default behavior
        mode, agent, agent_nr, path, small_reward, large_reward, learning_rate, n_simulations, state_choice, state_bin_size, use_rewardshaping = (
            'train', 
            'QAgent', 
            '1', 
            '', 
            10000, 
            30000, 
            0.01, 
            10, 
            ['storage_level', 'price', 'hour', 'Day_of_Week'],
            [10, 10, 24, 7]    
        )
        locals()[mode](agent, agent_nr, path, small_reward, large_reward, learning_rate, n_simulations, state_choice, state_bin_size, use_rewardshaping)
