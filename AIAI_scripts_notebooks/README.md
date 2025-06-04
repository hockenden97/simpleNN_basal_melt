# A summary of the relevant processing steps for NN emulators, and which files to use to do them

1) Process the *raw* NEMO files to get files with only the relevant fields including propagated temperature and salinity profiles
   This can be done in notebook using the files **Process_OPM0261.ipynb** and **Process_OPM0263.ipynb**, or in the terminal using **NN_prep_input.py**, **NN_prep_input_OPM0261.py**, **NN_prep_input_OPM0263.py** and the bash scripts **JOB_prep_OPM0263.sh** and **JOB_prep_OPM0261.sh**.
   Could be improved:
   * Create one file that will read all different simulation runs and process accordingly?
   * I think there might also be a problem with the slopes that has to be retro fitted in, but could ideally be added to this step!

2) Prepare the raw data into a format which can be used for training and testing the NN. 
     