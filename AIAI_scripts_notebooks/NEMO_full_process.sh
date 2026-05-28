#!/bin/bash

nemo_input="/bettik/ockendeh/NEMO_simulations/TGCC/eORCA1.L75-TOTO_y1985.1y_gridT.nc"
nemo_run="DUM001"
year="9999"
filepath_output="/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/data/processing_ho/"
nemo_output=${filepath_output}nn_output_melt_${nemo_run}_${year}.nc
job_type='NEMO'
echo $job_type $nemo_run $year
echo "Running these variables"

# Where to find the python script to run the job on 
path_python=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_scripts_notebooks/NEMO_full_process.py
path_jobid=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_scripts_notebooks/JOB_files/
path_local=JOB_files/

# Where to save the job output
path_jobname=$path_${job_type}_${nemo_run}_${year}

path_sh_file=${path_jobid}${job_type}_${nemo_run}_${year}
path_sh_local=${path_local}${job_type}_${nemo_run}_${year}

# Define the job that will run (load environment, save python output to log file) 
cat <<EOF > $path_sh_file.sh
#!/bin/bash
# Load the conda environment
# conda init
conda activate nnets_py38
# Run python with the specified variables 
# The 2>&1 means that errors in the python file will appear in the stdout file not the stderr file (I think) 
echo "Beginning python script"
python -u $path_python --nnin $nemo_input --nnout $nemo_output 2>&1 
echo 'Finished' $OAR_JOB_ID 
echo 'Finished' $OAR_JOB_ID 1>&2
EOF

# Make the job file executable
chmod +x $path_sh_file.sh

# And then execute it 
oarsub -S ./$path_sh_local.sh --stdout $path_jobid/$path_jobname.o --stderr $path_jobid/$path_jobname.e -l nodes=1/core=6,walltime=0:10:00 -n $path_jobname --project mais 
