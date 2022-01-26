cd ..
dir=logs/`date +%m%d`
mkdir -p $dir
nohup python main.py \
--dataset fmnist \
--from_csv niid1 \
--federated_type afl \
--drfa_gamma 0.01 \
> ${dir}/fmnist_niid1_afl_gamma001_`date +%H%M`.log &