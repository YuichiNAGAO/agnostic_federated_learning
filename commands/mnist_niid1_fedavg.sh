cd ..
dir=logs/`date +%m%d`
mkdir -p $dir
nohup python main.py \
--dataset mnist \
--from_csv niid1 \
--federated_type fedavg \
> ${dir}/mnist_niid1_fedavg_`date +%H%M`.log &