# UAV-TVT

This repository is created and shared for the following paper.
> Soltani, Nasim, et al. "RF Fingerprinting Unmanned Aerial Vehicles with Non-standard Transmitter Waveforms." IEEE Transactions on Vehicular Technology 69.12 (2020): 15518-15531. 
	https://ieeexplore.ieee.org/document/9277909/

> Dataset page: 	https://genesys-lab.org/hovering-uavs

## Steps to use the code:

1. Download the dataset from the above page.
2. Convert the dataset from SigMF format to .mat files using sigmf_reader.py
3. Preprocess the .mat files to generate train/validation/test partitions using pre_process_uav.py
4. Run the ML code using the bash script run_ML_code.sh (You need to edit the paths inside the .sh file.)
5. Aggregate the saved predictions using preds_aggregator.py

### Bash script run_ML_code.sh

	python -u /home/nasim/UAVFramework/ML_code/top.py \
	--exp_name $1 \
	--partition_path /home/nasim/UAV-TVT/PklFiles/cnn1/ \
	--stats_path /home/nasim/UAV-TVT/PklFiles/cnn1/ \
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
