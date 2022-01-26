cd ..
dir=logs/`date +%m%d`
mkdir -p $dir
nohup python main.py --dataset fmnist --from_csv iid --federated_type afl > ${dir}/fmnist_iid_afl_`date +%H%M`.log &