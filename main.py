from env import DataCenterEnv
from agent import QAgent, Heuristic
from Utils.args import parse_arguments
import sys
import os 

TRAIN = 'Data/train.xlsx'
VALIDATE = 'Data/validate.xlsx'

def train(agent, agent_nr, path, small_reward, large_reward, learning_rate, n_simulations, state_choice, state_bin_size):
    environment = DataCenterEnv(path or TRAIN)
    agent = globals()[agent](
        env=environment,
        agent_nr=agent_nr,
        small_reward=small_reward, 
        large_reward=large_reward, 
        learning_rate=learning_rate, 
        n_simulations=n_simulations,
        state_choice=state_choice,
        state_bin_size=state_bin_size
    )


    print(f"Training agent {agent_nr}...")
    agent.train()

def validate(agent, agent_nr, path):
    try:
        environment = DataCenterEnv(path or VALIDATE)
        agent = globals()[agent](
            env=environment,
            agent_nr=agent_nr
        )
        agent.evaluate()
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
        mode, agent, agent_nr, path, small_reward, large_reward, learning_rate, n_simulations, state_choice, state_bin_size = parse_arguments()
    else:
        mode, agent, agent_nr, path, small_reward, large_reward, learning_rate, n_simulations, state_choice, state_bin_size = (
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

    # Run
    if mode == 'train':
        locals()[mode](agent, agent_nr, path, small_reward, large_reward, learning_rate, n_simulations, state_choice, state_bin_size)
    elif mode == 'validate':
        locals()[mode](agent, agent_nr, path)
