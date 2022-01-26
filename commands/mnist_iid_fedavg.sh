cd ..
dir=logs/`date +%m%d`
mkdir -p $dir
nohup python main.py --dataset mnist --from_csv iid --federated_type fedavg > ${dir}/mnist_iid_fedavg_`date +%H%M`.log &