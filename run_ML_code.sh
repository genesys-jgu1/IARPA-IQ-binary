#!/bin/bash
# ----------------------------------------------------------------------------------------------------
python -u /home/nasim/UAVFramework/ML_code/top.py \
--exp_name $1 \
--partition_path /home/nasim/UAV-TVT/pkl_files/cnn1/ \
--stats_path /home/nasim/UAV-TVT/pkl_files/cnn1/ \
--save_path /home/nasim/UAV-TVT/results/ \
--model_flag alexnet \
--contin false \
--json_path /home/nasim/UAV-TVT/results/cnn1/model_file.json \
--hdf5_path /home/nasim/UAV-TVT/results/cnn1/model.hdf5 \
--slice_size 200 \
--num_classes 7 \
--batch_size 256 \
--id_gpu $2 \
--normalize true \
--train true \
--test true \
--epochs 100 \
--early_stopping true \
--patience 5 \
> /home/nasim/UAV-TVT/results/$1/log.out \
2> /home/nasim/UAV-TVT/results/$1/log.err
