import sys
import os
import argparse



def get_args():
    parser = argparse.ArgumentParser(description='Parameters for running training')
    parser.add_argument('--dataset',type=str, default='mnist',choices=['mnist','cifar10'],help='dataset name')
    parser.add_argument('--federated_type',type=str, default='fedavg',choices=['fedavg','afl'])
    parser.add_argument('--n_clients',type=int, default=3, help='the number of clients')
    parser.add_argument('--global_epochs',type=int, default=30, help='the number of global epochs')
    args = parser.parse_args()
    return args