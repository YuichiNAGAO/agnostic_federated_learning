

class LocalBase():
    def  __init__(self,args,dataset,val_dataset,client_id):
        self.args = args
        self.client_id = client_id
        

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
    