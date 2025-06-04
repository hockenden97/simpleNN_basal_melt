#!/bin/bash

mod_size='small'  #'mini', 'small', 'medium', 'large', 'extra_large'
TS_opt='extrap' # extrap, whole, thermocline
norm_method='std' # std, interquart, minmax
exp_name='slope_front'
#this_collection='OPM026_to2058' 
this_collection='OPM026_OPM031_10yr'
annual_f='' #  'annual_', or ''

# Where to find the python script to run the job on 
path_python=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_scripts_notebooks/batch_training.py
# Where to save the standardised job output files (.o and .e)
path_jobid=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_scripts_notebooks/TRAIN_files/
path_local=TRAIN_files/

for i in {1..10} # Set the seeds 1 to 10 for generating difference ensemble NN
do
    seed_nb=${i}     
    echo ${seed_nb}
    path_jobname=$path${mod_size}_${exp_name}_${this_collection}_${annual_f}${seed_nb}_${TS_opt}_${norm_method}
    echo "Running these variables: " $path_jobname
    path_sh_file=$path_jobid${mod_size}_${exp_name}_${this_collection}_${annual_f}${seed_nb}_${TS_opt}_${norm_method}
    path_sh_local=$path_local${mod_size}_${exp_name}_${this_collection}_${annual_f}${seed_nb}_${TS_opt}_${norm_method}
    
    # Define the job that will run (load environment, save python output to log file) 
    cat <<EOF > $path_sh_file.sh
    
    #!/bin/bash
    
    # Load the conda environment
    # conda init
    conda activate nnets_py38
    
    # Run python with the specified variables 
    # The 2>&1 means that errors in the python file will appear in the stdout file not the stderr file (I think) 
    python -u $path_python ${mod_size} ${TS_opt} ${norm_method} ${exp_name} ${seed_nb} ${this_collection}  ${annual_f} 2>&1 
    
    echo 'Finished' $OAR_JOB_ID 
    echo 'Finished' $OAR_JOB_ID 1>&2
EOF
    
    # Make the job file executable
    chmod +x $path_sh_file.sh
    
    # And then execute it 
    oarsub -S ./$path_sh_local.sh --stdout $path_jobid/$path_jobname.o --stderr $path_jobid/$path_jobname.e -l nodes=1/core=4,walltime=02:30:00 -n $path_jobname --project mais 
    
    # And then remove the sh file which runs the code because they clutter up the folder and are all just repeats 
    #rm $path_jobname.sh
done
