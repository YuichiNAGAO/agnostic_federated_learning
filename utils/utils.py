import copy
import torch
import numpy as np

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
