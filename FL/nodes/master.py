
import sys
sys.path.append('../../')
from utils.define_model import define_model
from utils.utils import weighted_average_weights
import pdb


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

    def aggregate(self,local_params):
        print("aggregating weights...")
        pdb.set_trace()


def define_globalnode(args):
    if args.federated_type=='fedavg':#normal
        return Fedavg_Global(args)
        
    elif args.federated_type=='afl':#afl
        return Afl_Global(args)
        
    else:       
        raise NotImplementedError     
        