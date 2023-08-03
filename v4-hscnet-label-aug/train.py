from __future__ import division

import sys
import os
import random
import argparse
from pathlib import Path
import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm

from datasets import get_dataset
from utils import *

#################################################################
parser = argparse.ArgumentParser(description="Hscnet")
parser.add_argument('--model', nargs='?', type=str, default='hscnet',
                    choices=('hscnet', 'scrnet'),
                    help='Model to use [\'hscnet, scrnet\']')
parser.add_argument('--dataset', nargs='?', type=str, default='7S', 
                    choices=('7S', '12S', 'i7S', 'i12S', 'i19S',
                    'Cambridge'), help='Dataset to use')
parser.add_argument('--scene', nargs='?', type=str, default='heads', 
                    help='Scene')
parser.add_argument('--n_iter', nargs='?', type=int, default=900000,
                    help='# of iterations (to reproduce the results from ' \
                    'the paper, 300K for 7S and 12S, 600K for ' \
                    'Cambridge, 900K for the combined scenes)')
parser.add_argument('--init_lr', nargs='?', type=float, default=1e-4,
                    help='Initial learning rate')
parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                    help='Batch size')
parser.add_argument('--resume', nargs='?', type=str, default=None,
                    help='Path to saved model to resume from')
parser.add_argument('--data_path', required=True, type=str, 
                    help='Path to dataset')
parser.add_argument('--log-summary', default='progress_log_summary.txt', 
                    metavar='PATH',
                    help='txt where to save per-epoch stats')
parser.add_argument('--train_id', nargs='?', type=str, default='',
                    help='An identifier string')
parser.add_argument('--scene_txt_postfix', nargs='?', type=str, default='', 
                    help='scene txt filename postfix')
args = parser.parse_args()

if args.dataset == '7S':
    if args.scene not in ['chess', 'heads', 'fire', 'office', 'pumpkin',
                          'redkitchen','stairs']:
        print('Selected scene is not valid.')
        sys.exit()

if args.dataset == '12S':
    if args.scene not in ['apt1/kitchen', 'apt1/living', 'apt2/bed',
                          'apt2/kitchen', 'apt2/living', 'apt2/luke', 
                          'office1/gates362', 'office1/gates381', 
                          'office1/lounge', 'office1/manolis',
                          'office2/5a', 'office2/5b']:
        print('Selected scene is not valid.')
        sys.exit()

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

#################################################################
# prepare datasets
if args.dataset in ['7S', 'i7S']: 
    dataset = get_dataset('7S')
if args.dataset in ['12S', 'i12S']: 
    dataset = get_dataset('12S')

dataset = dataset(args.data_path, args.dataset, args.scene,
                  model=args.model, scene_txt_postfix=args.scene_txt_postfix)

trainloader = data.DataLoader(dataset, batch_size=args.batch_size,
                                  num_workers=24, shuffle=False)

for _, (name) in enumerate(trainloader):
    continue
