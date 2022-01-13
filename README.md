# agnostic_federated_learning

Agnostic Federated Learning(2020 NIPS)の再現実装
https://arxiv.org/pdf/1902.00146.pdf


How to run
```
python main.py 
```

options:
```
  --dataset {mnist,cifar10}          
  --federated_type {fedavg,afl}     
  --model {cnn,mlp}         
  --n_clients int            
  --global_epochs int    
  --local_epochs int
  --batch_size int
  --on_cuda {yes,no}
  --optimizer {sgd,adam}
  --lr float
  --iid {yes,no}
  --drfa_gamma float
```
