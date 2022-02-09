# IARPA-IQ-binary
Work in progress. Version of Nasim Soltani's UAV-TVT code which intakes binary files.

Changes made:
* 'partition_path'->'test_label_path', fixed all correponding mentions
* 'loadmat'->'np.fromfile' (binary compatible)
* 'categorical_crossentropy'->'sparse_categorical_crossentropy' (TEST DATA ONLY, change back for real data)
* added input transposing
* changed syntax to be compatible with binary files

Steps to run are still the same:
1) Change dataset, pkl folder addresses; labels in preprocessing
2) Run preprocessing script
2) Make/change directories in bash file
4) Run bash file
