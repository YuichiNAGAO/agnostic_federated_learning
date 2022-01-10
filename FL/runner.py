
from .nodes.client import *
from .nodes.master import *

def runner_train(args, train_dataset, test_dataset,  epoch):
    
    if epoch==1:
        global clients
        global master
        clients={}
        for client_id in range(args.n_clients):
            clients[client_id]=define_localnode(args,train_dataset, test_dataset, client_id)
        master=define_globalnode(args)
    
    for client_id in clients.keys():
        clients[client_id]
        global_weight=master.distribute_weight()
        print(client_id)
        
    
    return