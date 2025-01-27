import argparse

def parse_train_arguments():
    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--agent', type=str, default='QAgent', choices=['QAgent', 'Heuristic'])
    args.add_argument('--agent_nr', type=str, default='1')
    args.add_argument('--path', type=str, default=None)
    args.add_argument('--small_reward', type=int, default=50000)
    args.add_argument('--large_reward', type=int, default=100000)
    args.add_argument('--learning_rate', type=float, default=0.05)
    args.add_argument('--n_simulations', type=int, default=10)
    args.add_argument('--state_choice', type=str, default='storage_level,price,hour')
    args.add_argument('--state_bin_size', type=str, default='10,10,24')
    args.add_argument('--use_rewardshaping', type=bool, default=True)
    args = args.parse_args()
    args.state_choice = args.state_choice.split(',')
    args.state_bin_size = args.state_bin_size.split(',')
    return (args.mode, args.agent, args.agent_nr, args.path, args.small_reward, 
            args.large_reward, args.learning_rate, args.n_simulations, 
            args.state_choice, args.state_bin_size, args.use_rewardshaping)

def parse_evaluate_arguments():
    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='validate')
    args.add_argument('--agent', type=str, default='QAgent', choices=['QAgent', 'Heuristic'])
    args.add_argument('--agent_nr', type=str, default='1')
    args.add_argument('--path', type=str, default=None)
    args.add_argument('--show_plot', type=bool, default=False)
    args.add_argument('--save_plot', type=bool, default=False)
    args.add_argument('--use_rewardshaping', type=bool, default=True)
    args = args.parse_args()
    return args.mode, args.agent, args.agent_nr, args.path, args.show_plot, args.save_plot, args.use_rewardshaping
