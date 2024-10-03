#!/bin/bash
#OAR -l nodes=1/core=3,walltime=04:00:00 
#OAR --project mais 

## RUN CROSS VALIDATION JOBS

mod_size=small  #'mini', 'small', 'medium', 'large', 'extra_large'
TS_opt=extrap # extrap, whole, thermocline
norm_method=std # std, interquart, minmax
exp_name=newbasic2
seed_nb=10

# The job script that corresponds to this job specifically 
path_jobscripts=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/BASH/

# Where the log of any written python files comes out 
path_outfiles=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/BASH/

# Where to find the python script to run the job on 
path_python=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/scripts_and_notebooks/training

# Where to save the job output
path_jobid=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/BASH/

path_jobname=${mod_size}_${TS_opt}_${norm_method}_wholedataset_${exp_name}_${seed_nb}
echo path_jobname

# Define the job that will run (load environment, save python output to log file) 
cat <<EOF > $path_jobscripts/${mod_size}_${TS_opt}_${norm_method}_wholedataset_${exp_name}_${seed_nb}.sh 

#!/bin/bash

conda activate nnets_py38
python -u $path_python/run_training_whole_dataset.py ${mod_size} ${TS_opt} ${norm_method} ${exp_name} ${seed_nb} 2>&1 | tee $path_outfiles/$path_jobname.log
EOF

chmod +x $path_jobscripts/${mod_size}_${TS_opt}_${norm_method}_wholedataset_${exp_name}_${seed_nb}.sh

oarsub -S -n $path_jobname --stdout $path_jobid/$path_jobname.o%jobid%  --stderr $path_jobid/$path_jobname.e%jobid% -p "network_address='luke62'" $path_jobscripts/$path_jobname.sh

# to remove if no CV!!!


