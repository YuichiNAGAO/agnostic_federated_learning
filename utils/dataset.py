
import copy
import torch
from torchvision import datasets, transforms
import pdb

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../dataset/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

    elif args.dataset == 'mnist':
        data_dir = '../dataset/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

    train_data_distribution = data_distribution(train_dataset, args.n_clients)
    test_data_distribution = data_distribution(test_dataset, args.n_clients)

    return train_dataset, test_dataset, train_data_distribution, test_data_distribution

def data_distribution(dataset, n_clients):
    idx_map={}
    for client_id in range(n_clients):
        idx=torch.where(dataset.targets == client_id)[0]
        idx_map[client_id]=idx
    print(idx_map)
    return idx_map