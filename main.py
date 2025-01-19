from env import DataCenterEnv
from agent import QAgent, Heuristic
from Utils.args import parse_arguments
import sys

TRAIN = 'Data/train.xlsx'
VALIDATE = 'Data/validate.xlsx'

def train(agent, path):
    environment = DataCenterEnv(path or TRAIN)
    agent = globals()[agent](environment)
    agent.train()

def validate(agent, path):
    environment = DataCenterEnv(path or VALIDATE)
    agent = globals()[agent](environment)
    agent.evaluate()


if __name__ == '__main__':

    # Command Line Support
    if len(sys.argv) > 1: mode, agent, path = parse_arguments()

    # Manual Selection
    else: mode, agent, path = ('train', 'QAgent', '')

    # Run
    locals()[mode](agent, path)
