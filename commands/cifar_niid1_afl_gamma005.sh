cd ..
dir=logs/`date +%m%d`
mkdir -p $dir
nohup python main.py \
--dataset cifar10 \
--from_csv niid1 \
--federated_type afl \
--drfa_gamma 0.05 \
> ${dir}/cifar_niid1_afl_gamma005_`date +%H%M`.log &