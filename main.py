from env import DataCenterEnv
from agent import QAgent, Heuristic
from Utils.args import parse_arguments
import sys

TRAIN = 'Data/train.xlsx'
VALIDATE = 'Data/validate.xlsx'

def train(agent, path, small_reward, large_reward, learning_rate, n_simulations, state_choice):
    environment = DataCenterEnv(path or TRAIN)
    agent = globals()[agent](
        environment, 
        small_reward=small_reward, 
        large_reward=large_reward, 
        learning_rate=learning_rate, 
        n_simulations=n_simulations,
        state_choice=state_choice
        )
    
    agent.train()

def validate(agent, path):
    environment = DataCenterEnv(path or VALIDATE)
    agent = globals()[agent](environment)
    agent.evaluate()

if __name__ == '__main__':
    # Command Line Support
    if len(sys.argv) > 1:
        mode, agent, path, small_reward, large_reward, learning_rate, n_simulations, state_choice = parse_arguments()
    else:
        mode, agent, path, small_reward, large_reward, learning_rate, n_simulations, state_choice = (
            'train', 
            'QAgent', 
            '', 
            10000, 
            30000, 
            0.01, 
            10, 
            ['storage_level', 'price', 'hour', 'day', 'Season']
            )

    # Run
    if mode == 'train':
        locals()[mode](agent, path, small_reward, large_reward, learning_rate, n_simulations, state_choice)
    elif mode == 'validate':
        locals()[mode](agent, path)
