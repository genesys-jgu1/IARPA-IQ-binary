#!/bin/bash
# ----------------------------------------------------------------------------------------------------
python -u /home/jerry/UAV-TVT-bin/ML_code/top.py \
--exp_name IQbin \
--test_label_path /home/jerry/IQ_pkl_files_bin/ \
--stats_path /home/jerry/IQ_pkl_files_bin/ \
--save_path /home/jerry/IQ_pkl_files_bin/results/ \
--model_flag alexnet \
--contin false \
--json_path '' \
--hdf5_path '' \
--slice_size 128 \
--num_classes 1 \
--batch_size 128 \
--id_gpu 0 \
--normalize true \
--train true \
--test true \
--epochs 30 \
--early_stopping true \
--patience 3 \
> /home/jerry/IQ_pkl_files_bin/results/log.out \
2> /home/jerry/IQ_pkl_files_bin/results/log.err
