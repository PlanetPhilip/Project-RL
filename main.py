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
state = environment.observation()
aggregate_reward = 0

# Rollout
terminated = False
while not terminated:
    action = agent.act(state)
    # next_state is given as: [storage_level, price, hour, day]
    next_state, reward, terminated = environment.step(action)
    state = next_state
    aggregate_reward += reward
    print("Action:", action)
    print("Next state:", next_state)
    print("Reward:", reward)

print('Total reward:', aggregate_reward)
