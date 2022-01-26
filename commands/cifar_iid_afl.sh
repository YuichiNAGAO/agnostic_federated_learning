cd ..
dir=logs/`date +%m%d`
mkdir -p $dir
nohup python main.py --dataset cifar10 --from_csv iid --federated_type afl > ${dir}/cifar_iid_afl_`date +%H%M`.log &