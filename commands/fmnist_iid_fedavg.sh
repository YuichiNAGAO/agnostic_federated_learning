cd ..
dir=logs/`date +%m%d`
mkdir -p $dir
nohup python main.py --dataset fmnist --from_csv iid --federated_type fedavg > ${dir}/fmnist_iid_fedavg_`date +%H%M`.log &