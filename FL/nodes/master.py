
import sys
sys.path.append('../../')
from utils.define_model import define_model

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


class Afl_Global(GlobalBase):
    def __init__(self, args):
        super().__init__(args)

    def aggregate(self,local_params):
        print("aggregating weights...")



def define_globalnode(args):
    if args.federated_type=='fedavg':#normal
        return Fedavg_Global(args)
        
    elif args.federated_type=='afl':#afl
        return Afl_Global(args)
        
    else:       
        raise NotImplementedError     
        