import argparse

def parse_arguments():
    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='validate', choices=['train', 'validate'])
    args.add_argument('--agent', type=str, default='QAgent', choices=['QAgent', 'Heuristic'])
    args.add_argument('--path', type=str, default=None)
    args = args.parse_args()
    return args.mode, args.agent, args.path
