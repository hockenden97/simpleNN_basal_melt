#!/bin/bash

exp_name=newbasic2

#OAR -n neural_net_test
#OAR --stdout wholedataset.o%jobid% 
#OAR --stderr wholedataset.e%jobid% 
#OAR -l nodes=1/core=3, walltime=04:00:00 
#OAR --project mais

mod_size=small  #'mini', 'small', 'medium', 'large', 'extra_large'
TS_opt=extrap # extrap, whole, thermocline
norm_method=std # std, interquart, minmax
seed_nb=10

conda activate nnets_py38

echo "Does this appear in the log?"

# The job script that corresponds to this job specifically 
path_jobscripts=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/BASH/

# Where the log of any written python files comes out 
path_outfiles=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/BASH/

# Where to find the python script to run the job on 
path_python=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/scripts_and_notebooks/training

# Where to save the job output
path_jobid=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/BASH/

path_jobname=${mod_size}_${TS_opt}_${norm_method}_wholedataset_${exp_name}_${seed_nb}
echo $path_jobname

python -u $path_python/run_training_whole_dataset.py ${mod_size} ${TS_opt} ${norm_method} ${exp_name} ${seed_nb} 2>&1 | tee $path_outfiles/$path_jobname.log