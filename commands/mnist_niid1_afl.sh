cd ..
dir=logs/`date +%m%d`
mkdir -p $dir
nohup python main.py --dataset mnist --from_csv niid1 --federated_type afl > ${dir}/mnist_niid1_afl_`date +%H%M`.log &