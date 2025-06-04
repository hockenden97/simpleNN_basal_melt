#!/bin/bash
#OAR -n nn_input_prep 
#OAR --stdout TRAIN_files/normalise2.o
#OAR --stderr TRAIN_files/normalise2.e 
#OAR -l nodes=1/core=4,walltime=02:30:00 
#OAR --project mais

# 13th December 2024

# conda init
conda activate nnets_py38
echo "Now python?"
python NORM_inputOPM.py 

echo "[end]"