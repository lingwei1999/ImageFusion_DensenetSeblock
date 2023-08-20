import warnings
warnings.filterwarnings('ignore')

import argparse
import logging
import torch
from argparse import Namespace

from pipeline.train import Train


from utils.environment_probe import EnvironmentProbe


def parse_args() -> Namespace:
    # args parser
    parser = argparse.ArgumentParser()

    # universal opt
    parser.add_argument('--id', default='test', help='train process identifier')
    parser.add_argument('--folder', default='../datasets/LLVIP', help='data root path')
    parser.add_argument('--size', default=256, help='resize image to the specified size')
    parser.add_argument('--cache', default='cache', help='weights cache folder')
    parser.add_argument('--load', type=str, default=None, help='epoch to train')
    # parser.add_argument('--load', type=str, default='./cache/default_maxWeightGT/best.pth', help='epoch to train')

    # checkpoint opt
    parser.add_argument('--epochs', type=int, default=8, help='epoch to train')
    # optimizer opt
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    # dataloader opt
    parser.add_argument('--batch_size', type=int, default=16, help='dataloader batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='dataloader workers number')

    # experimental opt
    parser.add_argument('--debug', action='store_true', help='debug mode (default: off)')

    return parser.parse_args()


if __name__ == '__main__':
    config = parse_args()
    logging.basicConfig(level='INFO')
    torch.cuda.empty_cache()
    torch.multiprocessing.set_start_method('spawn')

    environment_probe = EnvironmentProbe()
    train_process = Train(environment_probe, config)
    train_process.run()
