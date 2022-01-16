
import copy
import os 
import torch
from torchvision import datasets, transforms
import pdb
import numpy as np

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar10':
        data_dir = os.path.join(args.path,'dataset/')
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

    elif args.dataset == 'mnist':
        data_dir = os.path.join(args.path,'dataset/')

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

    train_data_distribution = data_distribution(train_dataset, args)
    test_data_distribution = data_distribution(test_dataset, args)

    return train_dataset, test_dataset, train_data_distribution, test_data_distribution

def data_distribution(dataset, args):
    idx_map={}
    n_clients=args.n_clients
    if args.iid:
        num_items = int(len(dataset)/n_clients)
        all_idxs = [i for i in range(len(dataset))]
        for i in range(n_clients):
            idx_map[i] = set(np.random.choice(all_idxs, num_items,replace=False))
            all_idxs = list(set(all_idxs) - idx_map[i])
        
    else: 
        for client_id in range(n_clients):
            if type(dataset.targets) is list:
                targets=torch.Tensor(dataset.targets)
            else:
                targets=dataset.targets
            # pdb.set_trace()
            idx=torch.where(targets == client_id)[0]
            idx_map[client_id]=idx
    return idx_map