from  dataclasses import dataclass
import numpy as np
from collections import OrderedDict


@dataclass
class ClientsParams:
    weight : OrderedDict = None
    afl_loss  : float = None