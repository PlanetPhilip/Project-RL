from env import DataCenterEnv
from agent import QAgent, Heuristic
from Utils.args import parse_train_arguments, parse_evaluate_arguments
import sys
import os

TRAIN = 'Data/train-cleaned-features.xlsx'
VALIDATE = 'Data/validate.xlsx'

def train(agent, agent_nr, path, small_reward, large_reward, learning_rate, n_episodes, state_bin_size, use_rewardshaping, show_plot=False, save_plot=False):
    environment = DataCenterEnv(path or TRAIN)
    agent = globals()[agent](
        env=environment,
        agent_nr=agent_nr,
        small_reward=small_reward, 
        large_reward=large_reward, 
        learning_rate=learning_rate, 
        n_episodes=n_episodes,
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
            mode, *args = parse_train_arguments()
            locals()[mode](*args)
        else:
            mode, *args = parse_evaluate_arguments()
            locals()[mode](*args)

    # Default behavior
    else:
        mode = 'validate'
        params = {
            'agent': 'Heuristic',
            'agent_nr': '1',
            'path': '',
            'small_reward': 10000,
            'large_reward': 30000,
            'learning_rate': 0.01,
            'n_episodes': 10,
            'state_bin_size': [10, 10, 24, 7, 4],
            'use_rewardshaping': True
        }
        locals()[mode](**params)
