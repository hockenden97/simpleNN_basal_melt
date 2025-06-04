import numpy as np
import xarray as xr    
import glob
import pandas as pd
import itertools
import sklearn
import os

# Create a function to read in the simulation runs and convert them to pd dataframes
def create_df_total(nemo_run, year, month, verbose = 0):
    ''' This function reads in a .nc xarray file, and adds in the mean and std of T and S '''
    ''' and the slope parameters, and then saves these as a pandas dataframe to be merged '''
    filepath_ij = filepath_nn_input + nemo_run + '_' + year + '_' + month + '.nc'
    if verbose == 1:
        print('You have loaded:', filepath_ij)
    data = xr.open_dataset(filepath_ij)
    mean_T = np.ones(basins_NEMO.shape)*np.nan
    mean_S = np.ones(basins_NEMO.shape)*np.nan
    std_T = np.ones(basins_NEMO.shape)*np.nan
    std_S = np.ones(basins_NEMO.shape)*np.nan
    for j in basin_nos:
        mask_basin = basins_NEMO*masks.closed_cavities_nan == j
        mean_T[mask_basin] = np.ones(len(data.temperature_prop.data[mask_basin])) * np.nanmean(data.temperature_prop.data[mask_basin])
        mean_S[mask_basin] = np.ones(len(data.temperature_prop.data[mask_basin])) * np.nanmean(data.salinity_prop.data[mask_basin])
        std_T[mask_basin] = np.ones(len(data.temperature_prop.data[mask_basin])) * np.nanstd(data.temperature_prop.data[mask_basin])
        std_S[mask_basin] = np.ones(len(data.temperature_prop.data[mask_basin])) * np.nanstd(data.salinity_prop.data[mask_basin])
    year_label = []
    month_label = []
    for i in range(len(std_S[closed_cavities == 1])):
        year_label.append(year)
        month_label.append(month)
    if np.sum(np.isnan(data.slope_is_lon.data[closed_cavities == 1])) > 0:
        df_total = pd.DataFrame({'lat': data.lat.data[closed_cavities == 1], 
                       'lon': data.lon.data[closed_cavities == 1], 
                       'distances_GL': data.distances_GL.data[closed_cavities == 1],
                       'distances_OO': data.distances_OO.data[closed_cavities == 1],
                       'distances_OC': data.distances_OC.data[closed_cavities == 1],
                       'temperature_prop': data.temperature_prop.data[closed_cavities == 1],
                       'salinity_prop': data.salinity_prop.data[closed_cavities ==1],
                       'melt_m_ice_per_y': data.melt_ice_per_yr.data[closed_cavities == 1],
                       'corrected_isdraft': data.corrected_isdraft.data[closed_cavities ==1],
                       'bathymetry': data.bathymetry.data[closed_cavities == 1],
                       'slope_is_lon': redone_slopes.slope_is_lon.data[closed_cavities == 1],
                       'slope_is_lat': redone_slopes.slope_is_lat.data[closed_cavities == 1],
                       'slope_ba_lon': redone_slopes.slope_ba_lon.data[closed_cavities == 1],
                       'slope_ba_lat': redone_slopes.slope_ba_lat.data[closed_cavities == 1],
                       'slope_is_across_front': redone_slopes.slope_is_across_front.data[closed_cavities == 1],
                       'slope_is_towards_front': redone_slopes.slope_is_towards_front.data[closed_cavities == 1],
                       'slope_ba_across_front': redone_slopes.slope_ba_across_front.data[closed_cavities == 1],
                       'slope_ba_towards_front': redone_slopes.slope_ba_towards_front.data[closed_cavities == 1],
                       'mean_T': mean_T[closed_cavities == 1],
                       'mean_S': mean_S[closed_cavities == 1],
                       'std_T': std_T[closed_cavities == 1],
                       'std_S': std_S[closed_cavities == 1], 
                       'year': year_label, 
                       'month': month_label, 
                       'basins_NEMO': basins_NEMO[closed_cavities == 1],
                       'area': areas.area.data[closed_cavities == 1] })
    else:
        df_total = pd.DataFrame({'lat': data.lat.data[closed_cavities == 1], 
                           'lon': data.lon.data[closed_cavities == 1], 
                           'distances_GL': data.distances_GL.data[closed_cavities == 1],
                           'distances_OO': data.distances_OO.data[closed_cavities == 1],
                           'distances_OC': data.distances_OC.data[closed_cavities == 1],
                           'temperature_prop': data.temperature_prop.data[closed_cavities == 1],
                           'salinity_prop': data.salinity_prop.data[closed_cavities ==1],
                           'melt_m_ice_per_y': data.melt_ice_per_yr.data[closed_cavities == 1],
                           'corrected_isdraft': data.corrected_isdraft.data[closed_cavities ==1],
                           'bathymetry': data.bathymetry.data[closed_cavities == 1],
                           'slope_is_lon': data.slope_is_lon.data[closed_cavities == 1],
                           'slope_is_lat': data.slope_is_lat.data[closed_cavities == 1],
                           'slope_ba_lon': data.slope_ba_lon.data[closed_cavities == 1],
                           'slope_ba_lat': data.slope_ba_lat.data[closed_cavities == 1],
                           'slope_is_across_front': data.slope_is_across_front.data[closed_cavities == 1],
                           'slope_is_towards_front': data.slope_is_towards_front.data[closed_cavities == 1],
                           'slope_ba_across_front': data.slope_ba_across_front.data[closed_cavities == 1],
                           'slope_ba_towards_front': data.slope_ba_towards_front.data[closed_cavities == 1],
                           'mean_T': mean_T[closed_cavities == 1],
                           'mean_S': mean_S[closed_cavities == 1],
                           'std_T': std_T[closed_cavities == 1],
                           'std_S': std_S[closed_cavities == 1], 
                           'year': year_label, 
                           'month': month_label, 
                           'basins_NEMO': basins_NEMO[closed_cavities == 1],
                           'area': areas.area.data[closed_cavities == 1] })
    df_total2 = df_total[~np.isnan(df_total['temperature_prop'])]
    return df_total2

# Set the nemo_run
nemo_run = 'OPM0263'
print('Nemo run:', nemo_run)

# Set all the filepaths that will be required 
filepath_data_ho = "/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/data/processing_ho/"
filepath_masks = filepath_data_ho + "masks_"
filepath_nico_on_nemo = filepath_data_ho + "nico_on_nemo_" + nemo_run + '.nc'
filepath_mask_nemo_run = filepath_masks + nemo_run + '.nc'
filepath_nn_input = filepath_data_ho + "nn_input_"
filepath_slopes = filepath_nn_input + nemo_run + '_' + 'redo_slopes2' + '.nc'
filepath_areas = "/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_data/Masks/areas.nc"

# Set where you would like to save the data 
data_out_fp = '/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_data/Training_data/'
# Create a dataframe with all the data which is saved in this location
intermediate_filepath = data_out_fp + nemo_run + '_' + 'whole_dataset' + '_' + 'not_yet_normalised.csv'
# Creat a dataframe with only the specified data (which is also saved in this location) 
#this_collection = 'no_mar_oct'
#fp_metrics = data_out_fp + this_collection + '_' + 'metrics_norm.nc'
#fp_var_train_norm = data_out_fp + this_collection + '_' + 'train_data.nc'
#fp_var_val_norm = data_out_fp + this_collection + '_' + 'val_data.nc'

# Check which simulation datasets are available to look at 
processed_files = glob.glob(filepath_nn_input + nemo_run + '_1*.nc') + glob.glob(filepath_nn_input + nemo_run + '_2*.nc')
print('There are', len(processed_files), 'datasets available')

years = []
months = []
for i in range(len(processed_files)):
#for i in range(10):
    path_sec = os.path.split(processed_files[i])[1].split('_')
    years.append(path_sec[3])
    months.append(path_sec[4].split('.')[0])
unique_years = np.unique(years)
unique_months = np.unique(months)
print('The available years are', unique_years)
print('The available months are', unique_months)
unique_years_int = np.ndarray(len(unique_years))
for i in range(len(unique_years)):
    unique_years_int[i] = int(unique_years[i])

# Load in the closed_cavitites mask, and the slopes 
masks = xr.open_dataset(filepath_mask_nemo_run)
closed_cavities = masks.closed_cavities.data
masks.close()
print('You have loaded:')
print(filepath_mask_nemo_run)
# Load in the slopes too 
if nemo_run == 'OPM026':
    redone_slopes = xr.open_dataset(filepath_slopes)
    print(filepath_slopes)
areas = xr.open_dataset(filepath_areas)
print(filepath_areas)

# Load the pre-interpolated grids from file
ds_load = xr.open_dataset(filepath_nico_on_nemo)
print('You have loaded:')
print(filepath_nico_on_nemo)
dsREDCAV_NEMO = ds_load.dsREDCAV_NEMO.data
basins_NEMO = ds_load.basins_NEMO.data
ds_load.close()
# Some ice shelves which should potentially be joined together into one bigger ice shelf
join_ice_shelves = True
if join_ice_shelves == True:
    # Dotson and Crosson?
    #basins_NEMO[basins_NEMO == 101] = 129
    # Abbot Ice Shelf
    basins_NEMO[basins_NEMO == 109] = 143
    # George VI
    basins_NEMO[basins_NEMO == 112] = 125
    # Lambert 
    basins_NEMO[basins_NEMO == 20] = 103
basin_nos_temp = np.unique(basins_NEMO)
count = np.zeros(len(basin_nos_temp))
for i in range(len(basin_nos_temp)):
    count[i] = np.sum((basins_NEMO*masks.closed_cavities_nan) == basin_nos_temp[i])
mask_keep_nos = count != 0
basin_nos = basin_nos_temp[mask_keep_nos]

print('You will save output to:')
print(intermediate_filepath)
merge = True
if merge == True:
    years_to_merge = unique_years_int
    months_to_merge = unique_months 
    for i,j in itertools.product(range(len(years_to_merge)), range(len(months_to_merge))):
        if i + j < 1:
            year = str(int(years_to_merge[0]))
            month = months_to_merge[0]
            df_total = create_df_total(nemo_run, year, month)
        else:
            year = str(int(years_to_merge[i]))
            month = months_to_merge[j]
            df_ij = create_df_total(nemo_run, year, month)
            df_total2 = pd.concat([df_total, df_ij], ignore_index=True)
            df_total = df_total2
        print((i * len(months_to_merge)) + j + 1, 'out of', (len(years_to_merge)* len(months_to_merge)), 'processed', end = '\r')   

    intermediate_save = True
    if intermediate_save == True:
        df_total.to_csv(intermediate_filepath, index = False)
        print('You have saved:                  ')
        print(intermediate_filepath)