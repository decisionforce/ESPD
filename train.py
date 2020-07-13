import gym
import torch

from args import get_args
from model import ActorCritic
from espd import espd


def main():
    args = get_args()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    env = gym.make(args.env_name)
    num_inputs = env.observation_space.spaces['observation'].shape[
        0] + env.observation_space.spaces['desired_goal'].shape[
            0]  # extended state
    num_actions = env.action_space.shape[0]
    network = ActorCritic(num_inputs, num_actions, layer_norm=args.layer_norm)
    network.to(device)
    
    '''joint train'''
    reward_record = []
    for i in range(args.num_parallel_run):
        args.seed += 1
        reward_record.append(espd(args, network, device))

    
if __name__ == '__main__':
    main()
