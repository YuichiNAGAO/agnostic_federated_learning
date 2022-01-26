import sys
import os
import numpy as np
import torch
import time
import random
from tqdm import tqdm
import pdb
import os 

from FL.utils.dataset import get_dataset
from FL.utils.utils import set_global_seeds, device_check
from FL.runner import runner_train
from FL.parameters import get_args

def main(args):
    
    
    #load dataset and split dataset to local clients
    train_dataset, test_dataset, args.train_distributed_data,  args.test_distributed_data= get_dataset(args)
    
    # #training
    for epoch in range(args.global_epochs):
        runner_train(args, train_dataset, test_dataset, epoch+1)
    
if __name__ == '__main__':
    
    args = get_args()
    device_check(args.on_cuda)
    for key , value in args._get_kwargs():
        print(f"{key}: {value}")
    args.root_path = os.getcwd()

    if args.seed:
        set_global_seeds(args.seed_num)

    main(args)