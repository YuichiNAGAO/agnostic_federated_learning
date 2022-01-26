
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
    data_dir = os.path.join(args.path,'dataset/')
    if args.dataset == 'cifar10':
        
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

    elif args.dataset == 'mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

    elif args.dataset == 'fmnist':

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=apply_transform)

        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=apply_transform)


    else:
        NotImplementedError
    
    train_data_distribution = data_distribution(train_dataset, args)
    test_data_distribution = data_distribution(test_dataset, args)

    return train_dataset, test_dataset, train_data_distribution, test_data_distribution

def data_distribution(dataset, args):
    idx_map={}
    n_clients=args.n_clients
    if args.iid=="iid":
        print("iid")
        num_items = int(len(dataset)/n_clients)
        all_idxs = [i for i in range(len(dataset))]
        for i in range(n_clients):
            idx_map[i] = set(np.random.choice(all_idxs, num_items,replace=False))
            all_idxs = list(set(all_idxs) - idx_map[i])
        
    elif args.iid=="one":
        print("one")
        for client_id in range(n_clients):
            if type(dataset.targets) is list:
                targets=torch.Tensor(dataset.targets)
            else:
                targets=dataset.targets
            # pdb.set_trace()
            idx=torch.where(targets == client_id)[0]
            idx=[int(i) for i in idx]
            idx_map[client_id]=set(idx)

    elif args.iid=="noniid":
        print("noniid")
        idx_map=noniid(dataset,n_clients)
    
    elif args.iid=="from_csv":
        idx_map=from_csv(dataset,args)

    else:
        NotImplementedError

    return idx_map
        

def noniid(dataset, num_users): 
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:vg
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    
    num_class=2
    num_shards=num_users*2
    num_imgs=len(dataset)//num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)
    

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # pdb.set_trace()
    return dict_users

def from_csv(dataset, args):
    csv_dir = os.path.join(args.path,'config',args.from_csv+'.csv')
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
    print(pth)
    config=np.loadtxt(pth,delimiter=',')
    return config