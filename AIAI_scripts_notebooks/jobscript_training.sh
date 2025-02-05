#!/bin/bash
#OAR -n nn_input_prep 
#OAR --stdout nn_training.o%jobid% 
#OAR --stderr nn_training.e%jobid% 
#OAR -l nodes=1/core=4,walltime=02:30:00 
#OAR --project mais

# 13th December 2024

conda activate nnets_py38
python training.py 

echo "[end]"