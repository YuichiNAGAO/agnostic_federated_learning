cd ..
dir=logs/`date +%m%d`
mkdir -p $dir
nohup python main.py --dataset cifar10 --from_csv iid --federated_type fedavg > ${dir}/cifar_iid_fedavg_`date +%H%M`.log &