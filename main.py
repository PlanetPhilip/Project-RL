from env import DataCenterEnv
from agent import QAgent
import numpy as np
import argparse

args = argparse.ArgumentParser()
args.add_argument('--path', type=str, default='Data/train.xlsx')
args = args.parse_args()

np.set_printoptions(suppress=True, precision=2)
path_to_dataset = args.path

# Initialize
environment = DataCenterEnv(path_to_dataset)
agent = QAgent(environment)

# Train Agent
# agent.train()

# Evaluate Agent
agent.evaluate()
