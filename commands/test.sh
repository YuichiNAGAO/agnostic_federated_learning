cd ..
dir=logs/`date +%m%d`/
echo $dir
echo ${dir}`date +%H%M`.log
mkdir -p $dir
# nohup python -c 'print("hello")' > ${dir}`date +%H%M`.log &