from scipy.io import loadmat, savemat
import glob
import numpy as np
import os
import json
from tqdm import tqdm

""" This script converts the binary files in path "bin_path" to .mat files in path 
    "mat_path" and chops each signal (example) into K=10 parts (sub-ex) """

""" To use this script, first download the dataset from the dataset page:
    https://genesys-lab.org/hovering-uavs, unzip it, and put the UAV-Sigmf-float
    folder in a folder named UAV-TVT. Adjust bin_path and mat_path below, accordingly."""

bin_path = '/home/nasim/UAV-TVT/UAV-Sigmf-float16/'
mat_path = '/home/nasim/UAV-TVT/mat_files/'

# if the mat_path does not exist, create it
if not os.path.isdir(mat_path):
    os.mkdir(mat_path)

all_files = glob.glob(bin_path + '*') # all data and meta-data files

for this_bin_path in tqdm(all_files):
    if this_bin_path.endswith('.bin'):   # we have a data file
        with open (this_bin_path,'rb') as handle:
            iq_seq = np.fromfile(handle, dtype='<f2')    # read little endian (<) I/Q samples value (f4 => float32)
        n_samples = iq_seq.shape[0]/2
        IQ_data = np.zeros((n_samples,2),dtype=np.float32)
        IQ_data[:,0] = iq_seq[range(0, iq_seq.shape[0]-1, 2)]    # load all I-values in dimension 0
        IQ_data[:,1] = iq_seq[range(1, iq_seq.shape[0], 2)]      # ...and Q-values in dimension 1

        # chop this example into K=10 sub-examples and save them individually
        sub_ex_len = int(IQ_data.shape[0]/10)
        start_index = 0
        for i in range(10):
            this_sub_ex = IQ_data[start_index:start_index+sub_ex_len, :]
            
            # reshape the sub-ex for the framework
            this_sub_ex = np.expand_dims(np.transpose(this_sub_ex[:,0]+1j*this_sub_ex[:,1]),axis=0)
            
            # save the sub-ex
            mat_name = this_bin_path.split('/')[-1].split('.')[0]+'_'+str(i)+'.mat'
            this_dict = {}
            this_dict['f_sig'] = this_sub_ex    # the mat file should be saved under the key 'f_sig' for the ML code to read it.
            savemat(os.path.join(mat_path,mat_name), this_dict)

            # adjust start index to save the next sub_ex
            start_index += sub_ex_len
