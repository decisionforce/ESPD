import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='ESPD')

    parser.add_argument('--env-name',
                        default='FetchPush-v1',
                        help='environment name (default: FetchPush-v1)')
                        # 'FetchPush-v1', 'FetchSlide-v1', 'FetchPickAndPlace-v1'
    parser.add_argument('--seed',
                        type=int,
                        default=1234,
                        help='seed (default: 1234)')
    parser.add_argument('--num-episode',
                        type=int,
                        default=600,
                        help='number of episode to run (default: 600)')
    parser.add_argument('--batch-size',
                        type=int,
                        default=2500,
                        help='batch size (default: 2500)')
    parser.add_argument('--max-step-per-round',
                        default=200,
                        type=int,
                        help='maximum number of steps (default: 200)')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.995,
                        help='descount factor (default: 0.995)')
    parser.add_argument('--log-num-episode',
                        type=float,
                        default=1,
                        help='print log after number of episodes (default: 1)')
    parser.add_argument('--num-epoch',
                        type=int,
                        default=30,
                        help='number of epoch (default: 30)')
    parser.add_argument('--minibatch-size',
                        type=int,
                        default=25,
                        help='batch size (default: 25)')
    parser.add_argument('--clip',
                        type=float,
                        default=0.2,
                        help='clip gradient (default: 0.2)')
    parser.add_argument('--factor', default=1.5,
                        help='I have no idea')
    parser.add_argument('--lr-hid', default=3e-4,
                        help='no idea either')
    parser.add_argument('--Horizon-max',
                        default=8,
                        help='maximum horizon in PCHID')
    parser.add_argument('--reward-pos', default=0.,
                        help='reward for success')

    parser.add_argument('--num-parallel-run',
                        default=1,
                        help='number of parallel to train')
    
    # tricks
    parser.add_argument('--layer-norm',
                        default=True,
                        help='layer normalization (default: True)')
    parser.add_argument('--replay-buffer-size-IER',
                        default=100000,
                        help='replay buffer size (default: 100000)')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    return args
