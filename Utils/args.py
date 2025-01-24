import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--agent', type=str, default='QAgent')
    parser.add_argument('--agent_nr', type=str, default='1')
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--small_reward', type=float, default=10000)
    parser.add_argument('--large_reward', type=float, default=30000)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--n_simulations', type=int, default=10)
    parser.add_argument('--state_choice', type=str, default="storage_level,price,hour,Day_of_Week")
    parser.add_argument('--state_bin_size', type=str, default="10,10,24,7")
    
    args = parser.parse_args()
    
    # Convert string lists to actual lists
    state_choice = args.state_choice.split(',')
    state_bin_size = list(map(int, args.state_bin_size.split(',')))
    
    return (args.mode, args.agent, args.agent_nr, args.path, args.small_reward, 
            args.large_reward, args.learning_rate, args.n_simulations, 
            state_choice, state_bin_size)
