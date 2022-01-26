
import sys
sys.path.append('../../')
from FL.utils.define_model import define_model
from FL.utils.utils import weighted_average_weights, euclidean_proj_simplex
import pdb
import torch

class GlobalBase():
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if args.on_cuda else 'cpu'
        arch=define_model(args.model)
        self.model=arch(args).to(self.device)
    
    def distribute_weight(self):
        return self.model


class Fedavg_Global(GlobalBase):
    def __init__(self, args):
        super().__init__(args)

    def aggregate(self,local_params):
        print("aggregating weights...")
        global_weight=self.model
        local_weights=[]
        for client_id ,dataclass in local_params.items():
            local_weights.append(dataclass.weight)
        w_avg=weighted_average_weights(local_weights,global_weight.state_dict())

        self.model.load_state_dict(w_avg)


class Afl_Global(GlobalBase):
    def __init__(self, args):
        super().__init__(args)
        self.lambda_vector= torch.Tensor([1/args.n_clients for _ in range(args.n_clients)])
        
    

    def aggregate(self,local_params):
        print("aggregating weights...")
        global_weight=self.model
        local_weights=[]
        lambda_vector=self.lambda_vector
        loss_tensor = torch.zeros(self.args.n_clients)
        for client_id ,dataclass in local_params.items():
            loss_tensor[client_id]=torch.Tensor([dataclass.afl_loss])
            local_weights.append(dataclass.weight)

        lambda_vector += self.args.drfa_gamma * loss_tensor
        lambda_vector=euclidean_proj_simplex(lambda_vector)
        lambda_zeros = lambda_vector <= 1e-3
        if lambda_zeros.sum() > 0:
            lambda_vector[lambda_zeros] = 1e-3
            lambda_vector /= lambda_vector.sum()
        self.lambda_vector=lambda_vector
        w_avg=weighted_average_weights(local_weights,global_weight.state_dict(),lambda_vector.to(self.device))
        print("lambda:",lambda_vector)
        self.model.load_state_dict(w_avg)


def define_globalnode(args):
    if args.federated_type=='fedavg':#normal
        return Fedavg_Global(args)
        
    elif args.federated_type=='afl':#afl
        return Afl_Global(args)
        
    else:       
        raise NotImplementedError     
        