from env import DataCenterEnv
from agent import QAgent, Heuristic
import numpy as np
import argparse

TRAIN = 'Data/transformed_dataset.xlsx'
VALIDATE = 'Data/validate.xlsx'

args = argparse.ArgumentParser()
args.add_argument('--path', type=str, default=TRAIN)
args = args.parse_args()

np.set_printoptions(suppress=True, precision=2)
path_to_dataset = args.path

# Train
environment = DataCenterEnv(TRAIN)
agent = QAgent(environment)
agent.train()

# Evaluate
#environment = DataCenterEnv(VALIDATE)
#agent = QAgent(environment)
#agent = Heuristic(environment)
#agent.evaluate()