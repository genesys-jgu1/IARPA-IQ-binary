#!/bin/bash
# ----------------------------------------------------------------------------------------------------
python -u /home/jerry/UAV-TVT-bin/ML_code/top.py \
--exp_name IQbin \
--data_path /home/jerry/tim/ \
--stats_path /home/jerry/IQ_pkl_files/ \
--save_path /home/jerry/IQ_pkl_files_bin/results/ \
--model_flag alexnet \
--contin true \
--json_path '/home/jerry/IQ_pkl_files/results/IQ/model_file.json' \
--hdf5_path '/home/jerry/IQ_pkl_files/results/IQ/model.hdf5' \
--slice_size 256 \
--num_classes 2 \
--batch_size 256 \
--id_gpu 0 \
--normalize true \
--train false \
--test true \
--epochs 30 \
--early_stopping true \
--patience 3 \
> /home/jerry/IQ_pkl_files_bin/results/log.out \
2> /home/jerry/IQ_pkl_files_bin/results/log.err
