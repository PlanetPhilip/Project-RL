import argparse

def parse_arguments():
    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='validate', choices=['train', 'validate'])
    args.add_argument('--agent', type=str, default='QAgent', choices=['QAgent', 'Heuristic'])
    args.add_argument('--path', type=str, default=None)
    args.add_argument('--small_reward', type=int, default=50000)
    args.add_argument('--large_reward', type=int, default=100000)
    args.add_argument('--learning_rate', type=float, default=0.05)
    args.add_argument('--n_simulations', type=int, default=10)
    args.add_argument('--state_choice', type=str, default='storage_level,price,hour,day,Season')
    args = args.parse_args()
    args.state_choice = args.state_choice.split(',')
    return args.mode, args.agent, args.path, args.small_reward, args.large_reward, args.learning_rate, args.n_simulations, args.state_choice
