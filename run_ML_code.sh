#!/bin/bash
# ----------------------------------------------------------------------------------------------------
python -u /home/nasim/UAVFramework/ML_code/top.py \
--exp_name $1 \
--partition_path /home/nasim/UWBDataSet/PklFiles/Day11_2m_all_orientations/ \
--stats_path /home/nasim/UWBDataSet/PklFiles/Day11_2m_all_orientations/ \
--save_path /home/nasim/CleanFramework/results/ \
--model_flag uwb \
--contin false \
--json_path /home/nasim/CleanFramework/results/dummy/model_file.json \
--hdf5_path /home/nasim/CleanFramework/results/dummy/model.hdf5 \
--slice_size 32 \
--num_classes 4 \
--batch_size 256 \
--id_gpu $2 \
--normalize true \
--train true \
--test true \
--epochs 100 \
--early_stopping true \
--patience 5 \
> /home/nasim/CleanFramework/results/$1/log.out \
2> /home/nasim/CleanFramework/results/$1/log.err
