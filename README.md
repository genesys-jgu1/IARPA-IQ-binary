# IARPA-IQ-binary
Work in progress. Version of Nasim Soltani's UAV-TVT code which intakes binary I/Q files for classification/anomaly detection. Please contact gu.je, soltani.n, vi.chaudhary, or d.roy {@northeastern.edu} with any questions, thanks!

Runs on python2. Requires Keras, TensorFlow, glob, and tqdm, among other libraries. The most recent versions should work for all libraries; please send me a message if this is not the case.

To use, please do the following steps:
1) Download/clone the repo. You may move the model.hdf5, model_file.json, and stats.pkl files, though please keep track of their locations/paths.
2) Move all data to be tested in one folder. This is your <data_path>. This code was meant to be tested with a NEU_LTE_DSSS dataset containing Combined_LTE_DSSS and OnlyLTE frames.
3) Create a <save_path> where your results and predictions will go.
4) Open run_ML_code.sh and fill out the following arguments in <>. The other arguments should be left alone. The pre-existing run_ML_code.sh has examples for syntax comparisons.
~~~
python -u /home/<user>/IARPA-IQ-binary/ML_code/top.py \       # Fill in with your username and location of code folder, if necessary.
--exp_name <exp> \                                            # Experiment name
--data_path /home/<user>/<data_path>/ \                       # Path to the data to be tested.
--stats_path /home/<user>/<stats_path>/ \                     # Path to the stats.pkl used to determine predictions 
--save_path /home/<user>/<save_path>/ \                       # Path to the results and predictions
--model_flag alexnet \                                        # Architecture name, alexnet or resnet. Don't touch
--contin true \                                               # Test using an existing model/weights. Don't touch
--json_path '/home/<user>/<model_path>/model_file.json' \     # Path to the model file, used if contin is true
--hdf5_path '/home/<user>/<weights_path>/model.hdf5' \        # Path to the weights file, used if contin is true
--slice_size 256 \                                            # Input size to NN. Don't touch
--num_classes 2 \                                             # Number of classes for predictions. Don't touch for now
--batch_size 256 \                                            # Batch size for training. Don't touch
--id_gpu 0 \                                                  # GPU number to use to run training/testing. 0 should be default. Don't touch
--normalize true \                                            # Normalizes data for training/testing. Don't touch
--train false \                                               # Train the model. Don't touch for now, as the model has already been created
--test true \                                                 # Test the model. Don't touch
--epochs 30 \                                                 # Number of run-throughs for training. Don't touch
--early_stopping true \                                       # Allows training to stop early if no increase in prediction accuracy is detected. Don't touch
--patience 3 \                                                # Specifies number of epochs to wait to do early stopping in training. Don't touch
> /home/<user>/<save_path>/log.out \                          # Displays training progress and testing results.
2> /home/<user>/<save_path>/log.err                           # Displays error tracebacks and run progress.
~~~
5) Run the code with /run_ML_code.sh
6) If the run stops immediately, check the log.err file (go to its directory and use 'tail -f log.err') for any errors and address them if able, or report the message to gu.je@northeastern.edu.
7) You can check progress with log.err and log.out. The predictions will be saved in <save_path> in preds.pkl, and the accuracy will be reported in log.out as (slice, example) accuracy.
