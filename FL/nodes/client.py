
import co@py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import pdb


# from utils.utils import define_classification_model, softmax
# from utils.dataclass import ClientsParams



class CreateDataset(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalBase():
    def  __init__(self,args,train_dataset,test_dataset,client_id):
        self.args = args
        self.client_id = client_id

        self.traindata=self.create_dataset(train_dataset,args.train_distributed_data[client_id])
        self.testdata=self.create_dataset(test_dataset,args.train_distributed_data[client_id])

    def create_dataset(self,dataset):


class Fedavg_Local(LocalBase):
    def __init__(self,args,train_dataset,val_dataset,client_id):
        super().__init__(args,train_dataset,val_dataset,client_id)
    

class Afl_Local(LocalBase):
    def __init__(self,args,train_dataset,val_dataset,client_id):
        super().__init__(args,train_dataset,val_dataset,client_id)


     
def define_localnode(args,train_dataset,val_dataset,client_id):
    if args.federated_type=='fedavg':#normal
        return Fedavg_Local(args,train_dataset,val_dataset,client_id)
        
    elif args.federated_type=='afl':#afl
        return Afl_Local(args,train_dataset,val_dataset,client_id)

    else:       
        raise NotImplementedError     
    