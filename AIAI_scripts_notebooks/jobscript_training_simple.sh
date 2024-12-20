#!/bin/bash
#OAR -n nn_input_prep 
#OAR --stdout run_input_prep.o%jobid% 
#OAR --stderr run_input_prep.e%jobid% 
#OAR -l nodes=1/core=4,walltime=01:30:00 
#OAR --project mais

# 13th December 2024

conda activate nnets_py38
python training.py 

echo "[end]"