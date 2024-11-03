#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate UDA2

# for file in directory 
path=/home/raghav/Downloads/cluster
for file in $path/cfgs/*.yaml
do
    model=$(basename $file)
    model="${model%.*}"
    echo "Processing $model file..."
    log_file=$path/logs/${model}_eval.log
    touch $log_file
    script -c  "python test.py --cfg_file $path/cfgs/$model.yaml --batch_size 8 --ckpt $path/models/$model.pth" -a $log_file
 
done
