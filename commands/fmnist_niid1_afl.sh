cd ..
dir=logs/`date +%m%d`
mkdir -p $dir
nohup python main.py \
--dataset fmnist \
--from_csv niid1 \
--federated_type afl \
> ${dir}/fmnist_niid1_afl_`date +%H%M`.log &