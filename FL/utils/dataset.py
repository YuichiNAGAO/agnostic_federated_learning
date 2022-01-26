
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
    data_dir = os.path.join(args.root_path,'dataset/')
    if args.dataset == 'cifar10':
        
        # apply_transform = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=transform_train)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

    elif args.dataset == 'mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize((32, 32))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

    elif args.dataset == 'fmnist':

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize((32, 32))])

        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=apply_transform)

        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=apply_transform)

    else:
        NotImplementedError
    
    train_data_distribution = data_distribution(train_dataset, args)
    test_data_distribution = data_distribution(test_dataset, args)

    return train_dataset, test_dataset, train_data_distribution, test_data_distribution

def data_distribution(dataset, args):
    
    if args.data_dist=="iid":
        idx_map=iid(dataset, args)
    
    elif args.data_dist=="from_csv":
        idx_map=from_csv(dataset,args)

    else:
        NotImplementedError

    return idx_map

def iid(dataset, args):
    idx_map={}
    n_clients=args.n_clients
    num_items = int(len(dataset)/n_clients)
    all_idxs = [i for i in range(len(dataset))]
    for i in range(n_clients):
        idx_map[i] = set(np.random.choice(all_idxs, num_items,replace=False))
        all_idxs = list(set(all_idxs) - idx_map[i])

def from_csv(dataset, args):
    csv_dir = os.path.join(args.root_path,'config',args.from_csv+'.csv')
    dist_config=np.loadtxt(csv_dir, delimiter=',')
    dict_users = {i: np.array([]) for i in range(len(dist_config))}
    if type(dataset.targets) is list:
        targets=torch.Tensor(dataset.targets)
    else:
        targets=dataset.targets
    for cls, dist in enumerate(dist_config.T):
        idx=torch.where(targets == cls )[0].numpy()
        l=len(idx)
        ratio_list=dist/np.sum(dist)
        for i, ratio in enumerate(ratio_list):
            if i==len(ratio_list)-1:
                dict_users[i] = np.concatenate((dict_users[i], idx), axis=0)
                break
            rand_set=np.random.choice(idx, size=int(l*ratio), replace=False)
            idx = np.setdiff1d(idx, rand_set)
            dict_users[i] = np.concatenate((dict_users[i], rand_set), axis=0)
    return dict_users

def read_config(pth):
    config=np.loadtxt(pth,delimiter=',')
    return config