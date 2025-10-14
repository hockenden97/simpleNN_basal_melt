#!/bin/bash

nemo_run='OPM016' # Options are 'OPM016', 'OPM018', 'OPM021', 'ctrl94', 'isf94', 'isfru94'


# Where to find the python script to run the job on 
path_python=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_scripts_notebooks/PREP_ALL_TSmelt.py
# Where to save the job output
path_jobid=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_scripts_notebooks/JOB_files/
path_local=JOB_files/
path_jobname=$path_${nemo_run}
echo "Running these variables: " $path_jobname
echo "path" $path
path_sh_file=$path_jobid${nemo_run}_${year}
path_sh_local=$path_local${nemo_run}_${year}
echo "path_sh_file" $path_sh_file
# Define the job that will run (load environment, save python output to log file) 
cat <<EOF > $path_sh_file.sh
#!/bin/bash
# Load the conda environment
# conda init
conda activate nnets_py38
# Run python with the specified variables 
# The 2>&1 means that errors in the python file will appear in the stdout file not the stderr file (I think) 
python -u $path_python ${nemo_run} 2>&1 
echo 'Finished' $OAR_JOB_ID 
echo 'Finished' $OAR_JOB_ID 1>&2
EOF

# Make the job file executable
chmod +x $path_sh_file.sh

# And then execute it 
oarsub -S ./$path_sh_local.sh --stdout $path_jobid/$path_jobname.o --stderr $path_jobid/$path_jobname.e -l nodes=1/core=4,walltime=2:00:00 -n $path_jobname --project mais 

# And then remove the sh file which runs the code because they clutter up the folder and are all just repeats 
#rm $path_jobname.sh
