"""
Created on Wed Jan 25 17:41 2022

This script is to train a NN on the whole dataset

Author: Clara Burgard
"""

import numpy as np
import xarray as xr
import pandas as pd
from tqdm.notebook import trange, tqdm
import glob
import datetime
import time
import sys

import tensorflow as tf
from tensorflow import keras

import sys
sys.path.insert(1, '/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt')

#def get_model(size,shape,activ_fct,output_shape): #'mini', 'small', 'medium', 'large', 'extra_large'
#    model = keras.models.Sequential()
#    model.add(keras.layers.Input(shape, name="InputLayer"))
#    if size == 'small':
#        model.add(keras.layers.Dense(32, activation=activ_fct, name='Dense_n1'))
#        model.add(keras.layers.Dense(64, activation=activ_fct, name='Dense_n2'))
#        model.add(keras.layers.Dense(32, activation=activ_fct, name='Dense_n3'))    
#    model.add(keras.layers.Dense(output_shape, name='Output'))
#    model.compile(optimizer = 'adam',
#                  loss      = 'mse',
#                  metrics   = ['mae', 'mse'] ) 
#    return model
import nn_functions.model_functions as modf

######### READ IN OPTIONS

#mod_size = str(sys.argv[1]) #'mini', 'small', 'medium', 'large', 'extra_large'
#TS_opt = str(sys.argv[2]) # extrap, whole, thermocline
#norm_method = str(sys.argv[3]) # std, interquart, minmax
#exp_name = str(sys.argv[4])
#seed_nb = int(sys.argv[5])

mod_size = 'small'
TS_opt = 'extrap'
norm_method = 'std'
exp_name = 'NEMO_grid_v0_slope_front'
seed_nb = 1
print('Set options')

np.random.seed(seed_nb)
tf.random.set_seed(seed_nb)


if exp_name == 'onlyTSdraft':
    var_list = ['corrected_isfdraft','theta_in','salinity_in','melt_m_ice_per_y']
elif exp_name == 'TSdraftbotandiceddandwcd':
    var_list = ['corrected_isfdraft','theta_in','salinity_in','water_col_depth','theta_bot','salinity_bot','melt_m_ice_per_y']
elif exp_name == 'TSdraftbotandiceddandwcdreldGL':
    var_list = ['corrected_isfdraft','theta_in','salinity_in','water_col_depth','theta_bot','salinity_bot','rel_dGL',
                'melt_m_ice_per_y']
elif exp_name == 'onlyTSdraftandslope':
    var_list = ['corrected_isfdraft','theta_in','salinity_in','slope_ice_lon','slope_ice_lat','melt_m_ice_per_y']
elif exp_name == 'onlyTSdraft2':
    var_list = ['corrected_isfdraft','theta_in','salinity_in','melt_m_ice_per_y']
elif exp_name == 'TSTfdGLdIFwcd':
    var_list = ['corrected_isfdraft','theta_in','salinity_in','dGL','dIF','slope_ice_lon','slope_ice_lat','water_col_depth',
                'melt_m_ice_per_y']
elif exp_name == 'TSdraftslopereldGL':
    var_list = ['corrected_isfdraft','theta_in','salinity_in','slope_ice_lon','slope_ice_lat','rel_dGL','melt_m_ice_per_y']
elif exp_name == 'allbutconstants':
    var_list = ['dGL','dIF','corrected_isfdraft','bathy_metry','slope_bed_lon','slope_bed_lat','slope_ice_lon','slope_ice_lat',
                'isfdraft_conc','theta_in','salinity_in','u_tide','melt_m_ice_per_y']
elif exp_name == 'newbasic':
    var_list = ['dGL','dIF','corrected_isfdraft','bathy_metry','slope_bed_lon','slope_bed_lat','slope_ice_lon','slope_ice_lat','theta_in','salinity_in','melt_m_ice_per_y']
elif exp_name == 'newbasic2':
    var_list = ['dGL','dIF','corrected_isfdraft','bathy_metry','slope_bed_lon','slope_bed_lat','slope_ice_lon','slope_ice_lat','theta_in','salinity_in','T_mean', 'S_mean', 'T_std', 'S_std','melt_m_ice_per_y']
elif exp_name == 'NEMO_grid_v0_slope_lat_lon':
    var_list = ['distances_GL', 'distances_OO', 'distances_OC', 'temperature_prop', 'salinity_prop', 'corrected_isdraft', 'bathymetry', 'slope_is_lon', 'slope_is_lat', 'slope_ba_lon', 'slope_ba_lat', 'mean_T', 'mean_S', 'std_T', 'std_S', 'melt_ice_per_yr']
elif exp_name == 'NEMO_grid_v0_slope_front':
    var_list = ['distances_GL', 'distances_OO', 'distances_OC', 'temperature_prop', 'salinity_prop', 'corrected_isdraft', 'bathymetry', 'slope_is_across_front', 'slope_is_towards_front', 'slope_ba_across_front', 'slope_ba_towards_front', 'mean_T', 'mean_S', 'std_T', 'std_S', 'melt_ice_per_yr']

print('Set var list')

######### READ IN DATA

# Filepath for normalised inputs
filepath_data_AIAI = "/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_data/"
filepath_norm_data = filepath_data_AIAI + 'normed_data'
inputpath_data = filepath_norm_data + '_' 
# Filepath for outputs 
outputpath_nn_models = filepath_data_AIAI + 'NN_models/'
#outputpath_doc = '/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/custom_doc/experiments/'
#outputpath_doc = 

path_model = outputpath_nn_models

if TS_opt == 'extrap':
    
    data_train_orig_norm = xr.open_dataset(inputpath_data + 'train_data_wholedataset.nc')
    data_val_orig_norm = xr.open_dataset(inputpath_data + 'val_data_wholedataset.nc') 

    data_train_norm = data_train_orig_norm[var_list]
    data_val_norm = data_val_orig_norm[var_list]

    ## prepare input and target
    y_train_norm = data_train_norm['melt_ice_per_yr'].sel(norm_method=norm_method).load()
    x_train_norm = data_train_norm.drop_vars(['melt_ice_per_yr']).sel(norm_method=norm_method).to_array().load()
    
    y_val_norm = data_val_norm['melt_ice_per_yr'].sel(norm_method=norm_method).load()
    x_val_norm = data_val_norm.drop_vars(['melt_ice_per_yr']).sel(norm_method=norm_method).to_array().load()

else:
    print('Sorry, I dont know this option for TS input yet, you need to implement it...')

print('Loaded data')

######### TRAIN THE MODEL

#input_size = x_train_norm.values.shape[0]
input_size = x_train_norm.values.shape[0]
activ_fct = 'relu' 
epoch_nb = 100
batch_siz = 512

input_shape=(input_size,)

model = modf.get_model(mod_size, input_shape, activ_fct,1)

print('Loaded model')

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.0000001, min_delta=0.0005) #, min_delta=0.1
            
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    #min_delta=0.000001,
    patience=10,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)

time_start = time.time()
time_start0 = datetime.datetime.now()
print('Starting to fit model at:', time_start0)

history = model.fit(x_train_norm.T.values,
                    y_train_norm.values,
                    epochs          = epoch_nb,
                    batch_size      = batch_siz,
                    verbose         = 2,
                    validation_data = (x_val_norm.T.values, y_val_norm.values),
                   callbacks=[reduce_lr, early_stop])
time_end = time.time()
timelength = time_end - time_start

time_end0 = datetime.datetime.now()
print('Runtime:', timelength)
print('Also runtime?:', time_end0 - time_start0)
print()
print('The trained neural network is being saved to this location :')
print(path_model + 'model_nn_'+mod_size+'_'+exp_name+'_wholedataset_'+str(seed_nb).zfill(2)+'_TS'+TS_opt+'_norm'+norm_method+'.h5')
model.save(path_model + 'model_nn_'+mod_size+'_'+exp_name+'_wholedataset_'+str(seed_nb).zfill(2)+'_TS'+TS_opt+'_norm'+norm_method+'.h5')
print()

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

hist_csv_file = path_model + 'history_'+mod_size+'_'+exp_name+'_wholedataset_'+str(seed_nb).zfill(2)+'_TS'+TS_opt+'_norm'+norm_method+'.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
print('The history file has also been saved to csv.')