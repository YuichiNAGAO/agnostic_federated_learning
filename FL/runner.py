
from .nodes.client import *
from .nodes.master import *

def runner_train(args, train_dataset, test_dataset,  epoch):
    clients={}
    for client_id in range(args.n_clients):
        clients[client_id]=define_localnode(args,train_dataset, test_dataset, client_id)
    
    return