import numpy as np


class QAgent:

    def __init__(self, env, discount_rate=0.95, bin_size=20):
        self.env = env
        self.discount_rate = discount_rate
        self.action_space = self.env.continuous_action_space

        # Create bins
        self.bin_storage_level = np.linspace(env.storage_level, 170, bin_size)
        self.bin_price = np.linspace(np.min(env.price_values), np.max(env.price_values, bin_size))
        self.bin_hour = np.linspace(env.hour, 24, bin_size)
        self.bin_day = np.linspace(env.day, len(env.price_values), bin_size)
        self.bins = [self.bin_storage_level, self.bin_price, self.bin_hour, self.bin_day]

        # Construct Q-table
        discrete_state_space = (bin_size - 1,) * len(self.bins)  # TODO: do we need to do -1 ?
        discrete_action_space = (self.action_space,)  # TODO: discretize action space
        self.Qtable = np.zeros(discrete_state_space + discrete_action_space)


    def discretize_state(self):
        # TODO: write function
        return state

    def act(self, state):
        # TODO: write function
        action = np.random.uniform(-1, 1)
        return action


if __name__ == '__main__':

    # Create environment
    from env import DataCenterEnv

    environment = DataCenterEnv('./Data/train.xlsx')

    # Initialize agent
    agent = QAgent(environment)

    # Run agent
    aggregate_reward = 0
    terminated = False
    state = environment.observation()
    while not terminated:
        action = agent.act(state)
        next_state, reward, terminated = environment.step(
            action)  # next_state is given as: [storage_level, price, hour, day]
        state = next_state
        aggregate_reward += reward

        print("Action:", action, )
        print("Next state:", next_state)
        print("Reward:", reward)
    print('Total reward:', aggregate_reward)
