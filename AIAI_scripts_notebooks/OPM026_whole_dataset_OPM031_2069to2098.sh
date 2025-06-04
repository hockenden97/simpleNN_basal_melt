#!/bin/bash
# Load the conda environment
# conda init
conda activate nnets_py38
# Run python with the specified variables 
# The 2>&1 means that errors in the python file will appear in the stdout file not the stderr file (I think) 
echo "Beginning python script"
python -u /bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_scripts_notebooks/APPLY_trainedNNtoOPM.py OPM026_whole_dataset OPM031_2069to2098 slope_front  2>&1 
echo 'Finished'  
echo 'Finished'  1>&2
