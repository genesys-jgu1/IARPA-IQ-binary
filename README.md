# UAV-TVT

This repository is created and shared for the following paper.
> Soltani, Nasim, et al. "RF Fingerprinting Unmanned Aerial Vehicles with Non-standard Transmitter Waveforms." IEEE Transactions on Vehicular Technology 69.12 (2020): 15518-15531. 
	https://ieeexplore.ieee.org/document/9277909/

> Dataset page: 	https://genesys-lab.org/hovering-uavs

## Steps to use the code:

1. Download the dataset from the above page.
2. Convert the dataset from SigMF format to .mat files using sigmf_converter.py
3. Preprocess the .mat files to generate train/validation/test partitions using pre_process_uav.py
4. Run the ML code using the bash script run_ML_code.sh (You need to edit the paths inside the .sh file.)
5. Aggregate the saved predictions using preds_aggregator.py
