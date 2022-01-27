import sys
import os
import argparse


def strtobool (val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))


def get_args():
    parser = argparse.ArgumentParser(description='Parameters for running training')
    parser.add_argument('--dataset',type=str, default='cifar10',choices=['mnist','cifar10','fmnist'],help='dataset name')
    parser.add_argument('--federated_type',type=str, default='fedavg',choices=['fedavg','afl'])
    parser.add_argument('--model',type=str, default='cnn',choices=['cnn','mlp'])
    parser.add_argument('--n_clients',type=int, default=3, help='the number of clients')
    parser.add_argument('--global_epochs',type=int, default=30, help='the number of global epochs')
    parser.add_argument('--local_epochs',type=int, default=5, help='the number of local epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--on_cuda",  default="yes", type=strtobool)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=['sgd','adam'])
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument("--data_dist", type=str, default='from_csv',choices=['iid','from_csv'],help='local data distribution')
    parser.add_argument("--from_csv", type=str,  default='iid')
    parser.add_argument("--seed", default="yes", type=strtobool)
    parser.add_argument("--seed_num", default=0, type=int)
    # For AFL
    parser.add_argument('--drfa_gamma', default=0.01, type=float)

    args = parser.parse_args()
    return args