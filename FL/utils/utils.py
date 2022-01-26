import copy
import torch
import numpy as np
import pdb
import random

def weighted_average_weights(local_weights,global_weight,coff=None):
    """
    Returns the average of the weights.
    """
    if coff is None:
        coff=np.array([1/len(local_weights) for _ in range(len(local_weights))])
    w_avg = copy.deepcopy(global_weight)
    for key in w_avg.keys():
        for i in range(len(local_weights)):
            if w_avg[key].dtype==torch.int64:
                continue
            w_avg[key] += coff[i]*(local_weights[i][key]-global_weight[key])
    return w_avg


def euclidean_proj_simplex(v, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and (v >= 0).all():
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = torch.flip(torch.sort(v)[0],dims=(0,))
    cssv = torch.cumsum(u,dim=0)
    # get the number of > 0 components of the optimal solution
    non_zero_vector = torch.nonzero(u * torch.arange(1, n+1) > (cssv - s), as_tuple=False)
    if len(non_zero_vector) == 0:
        rho=0.0
    else:
        rho = non_zero_vector[-1].squeeze()
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w  

def set_global_seeds(seed_number):
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_number)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)

def device_check(on_cuda):
    if torch.cuda.is_available():
        print("GPU will be used for training\n")
    else:
        if on_cuda:
            message = "GPU is not available"
            raise ValueError(message)
        message = "Warning!: CPU will be used for training\n"
        print(message, flush=True)