import argparse
import os

import torch
import torch.multiprocessing as mp

import numpy as np
import helper
from train import work
from model import DFP_Network
from pycrayon import CrayonClient


OFFSETS = [1, 2, 4, 8, 16, 32]      # Set of temporal offsets



def __pars_args__():
    parser = argparse.ArgumentParser(description='DPF')

    parser.add_argument('--max_grad_norm', type=float, default=100, help='value loss coefficient (default: 100)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num_processes', type=int, default=4, help='how many training processes to use (default: 4)')
    parser.add_argument('--no_shared', default=False, help='use an optimizer without shared momentum.')

    parser.add_argument('-a_size', '--action_space', type=int, default=4, help='Action space size')
    parser.add_argument('-n_m', '--num_measurements', type=int, default=2, help='number of measurements in our enviroments')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--train', default=True, help='if we want to update the master weights')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='batch size used during learning')
    parser.add_argument('-o', '--offset', type=list, default=[1, 2, 4, 8, 16, 32], help='offsets using during training')


    parser.add_argument('-m_path', '--model_path', default='./model_goals', help='Path to save the model')
    parser.add_argument('-g_path', '--gif_path', default='./frames_goals', help='Path to save gifs of agent performance')

    parser.add_argument('--partial', default=False, help='if we want to use a partial enviroments')
    parser.add_argument('--env_size', type=int, default=5, help='size of our enviroments')

    return parser.parse_args()

if __name__ == '__main__':
    args = __pars_args__()

    master_net = DFP_Network((args.env_size**2)*3,                          # observation_size = (args.env_size*args.env_size)*3 = battel_ground*colors
                             num_offset=len(args.offset),
                             a_size=args.action_space,
                             num_measurements=args.num_measurements,
                             is_master=True)
    master_net.share_memory()
    cc = CrayonClient(hostname="localhost")
    # cc.remove_all_experiments()

    processes = []
    # p = mp.Process(target=work, args=(0, args, master_net, exp_buff, optimizer))      eval net
    # p.start()
    # processes.append(p)

    for rank in range(0, args.num_processes):
    # for rank in range(0, 1):
        p = mp.Process(target=work, args=(rank, args, master_net, cc, None))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()