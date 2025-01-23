from env import DataCenterEnv
from agent import QAgent, Heuristic
from Utils.args import parse_arguments
import sys

TRAIN = 'Data/train.xlsx'
VALIDATE = 'Data/validate.xlsx'

def train(agent, path, small_reward, large_reward, learning_rate, n_simulations):
    environment = DataCenterEnv(path or TRAIN)
    agent = globals()[agent](environment, small_reward=small_reward, large_reward=large_reward, learning_rate=learning_rate, n_simulations=n_simulations)
    agent.train()

def validate(agent, path):
    environment = DataCenterEnv(path or VALIDATE)
    agent = globals()[agent](environment)
    agent.evaluate()

if __name__ == '__main__':
    # Command Line Support
    if len(sys.argv) > 1:
        mode, agent, path, small_reward, large_reward, learning_rate, n_simulations = parse_arguments()
    else:
        mode, agent, path, small_reward, large_reward, learning_rate, n_simulations = ('train', 'QAgent', '', 50000, 100000, 0.05, 10)

    # Run
    if mode == 'train':
        locals()[mode](agent, path, small_reward, large_reward, learning_rate, n_simulations)
    elif mode == 'validate':
        locals()[mode](agent, path)
