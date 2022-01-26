cd ..
dir=logs/`date +%m%d`
mkdir -p $dir
nohup python main.py \
--dataset cifar10 \
--from_csv niid1 \
--federated_type fedavg \
> ${dir}/cifar_niid1_fedavg_`date +%H%M`.log &