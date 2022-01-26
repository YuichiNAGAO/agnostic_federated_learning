cd ..
dir=logs/`date +%m%d`
mkdir -p $dir
nohup python main.py --dataset cifar10 --from_csv niid1 --federated_type afl > ${dir}/cifar_niid1_afl_`date +%H%M`.log &