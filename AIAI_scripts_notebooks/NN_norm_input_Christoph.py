import numpy as np
import xarray as xr    
import glob
import pandas as pd
import itertools
import sklearn

# Set the nemo_run
nemo_run = 'Christoph'

where_are_you_running = 'bettik'#, 'bettik'
# Set the filepaths for the data 
if where_are_you_running == 'bettik':
    filepath_base = '/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/'
elif where_are_you_running == 'laptop':
    filepath_base = '/Users/ockendeh/Documents/simpleNN_basal_melt/'
else:
    print('Help: I don\'t know where to look for files')

# The input TSmelt files 
filepath_nn_input = filepath_base + 'AIAI_data/Christoph/TSmelt_v2'
# The geometry masks and geometry variables for this simulation
# A geometry file with the masks of the different types of cell and the basins 
filepath_mask_nemo_run = filepath_base +  'AIAI_data/Christoph/geometric_masks.nc'
# A geometry file with the required variables for the NN 
filepath_geomvars = filepath_base + 'AIAI_data/Christoph/geom_vars.nc'

# Set where you would like to save the data 
data_out_fp =  filepath_base + 'AIAI_data/Training_data/'

# Check which simulation datasets are available to look at 
processed_files = glob.glob(filepath_nn_input +'_*.nc')
print('There are', len(processed_files), 'datasets available')

import os
years = []
months = []
for i in range(len(processed_files)):
    path_sec = processed_files[i].split(filepath_nn_input)
    years.append(path_sec[1].split('_')[1].split('y')[1])
    months.append(path_sec[1].split('_')[2].split('.')[0].split('m')[1])
unique_years = np.unique(years)
unique_months = np.unique(months)
print('The available years are', unique_years)
print('The available months are', unique_months)
unique_years_int = np.ndarray(len(unique_years))
for i in range(len(unique_years)):
    unique_years_int[i] = int(unique_years[i])

def load_geom_files(filepath_geomvars, filepath_mask, join_ice_shelves = False):
    geoms = xr.open_dataset(filepath_geomvars)
    masks = xr.open_dataset(filepath_mask)
    print('You have loaded:')
    print(filepath_geomvars)
    print(filepath_mask)
    bN_00 = masks.basins_NEMO
    # Some ice shelves which should potentially be joined together into one bigger ice shelf
    if join_ice_shelves == True:
        # Dotson and Crosson?
        #basins_NEMO[basins_NEMO == 101] = 129
        # Abbot Ice Shelf
        bN_01 = xr.where(bN_00 == 109, 143, bN_00)
        # George VI
        bN_02 = xr.where(bN_01 == 112, 125, bN_01)
        # Lambert 
        basins_merged = xr.where(bN_02 == 20, 103, bN_02)
        print('Ice shelves joined, as requested')
    else:
        basins_merged = bN_00    
    return masks, basins_merged, geoms

# Load in the closed_cavitites mask, and the geometry files 
masks, basins_merged, geoms = load_geom_files(filepath_geomvars, filepath_mask_nemo_run, join_ice_shelves = True)
basin_nos_temp = np.unique(basins_merged)
count = np.zeros(len(basin_nos_temp))
for i in range(len(basin_nos_temp)):
    count[i] = np.sum((basins_merged*masks.closed_cavities_nan) == basin_nos_temp[i])
mask_keep_nos = count != 0
basin_nos = basin_nos_temp[mask_keep_nos]
print('There are', len(basin_nos), 'basins with data in.')

check_now_for_nans = False
if check_now_for_nans == True:
    for i in range(len(processed_files)):
        data = xr.open_dataset(processed_files[i])
        nansum = np.sum(np.isnan(data.temperature_prop.data[masks.closed_cavities == 1]))
        if nansum > 0:
            print(processed_files[i], nansum)

def create_df_total(year, month, masks, basins_merged, geoms):
    ''' This function reads in a .nc xarray file, and adds in the mean and std of T and S '''
    ''' and the slope parameters, and then saves these as a pandas dataframe to be merged '''
    filepath_ij = filepath_nn_input + '_y' + str(year) + '_m' + str(month).zfill(2) + '.nc' 
    data = xr.open_dataset(filepath_ij)
    if np.sum(np.isnan(data.temperature_prop.data[masks.closed_cavities ==1])) != 0:
        print('Nans present in data')
    mean_T = np.ones(masks.basins_NEMO.shape)*np.nan
    mean_S = np.ones(masks.basins_NEMO.shape)*np.nan
    std_T = np.ones(masks.basins_NEMO.shape)*np.nan
    std_S = np.ones(masks.basins_NEMO.shape)*np.nan
    for j in basin_nos:
        mask_basin = basins_merged*masks.closed_cavities_nan == j
        if np.sum(np.isnan(data.temperature_prop.data[mask_basin])) == 0:
            mean_T[mask_basin] = np.ones(len(data.temperature_prop.data[mask_basin])) * np.nanmean(data.temperature_prop.data[mask_basin])
            mean_S[mask_basin] = np.ones(len(data.temperature_prop.data[mask_basin])) * np.nanmean(data.salinity_prop.data[mask_basin])
            std_T[mask_basin] = np.ones(len(data.temperature_prop.data[mask_basin])) * np.nanstd(data.temperature_prop.data[mask_basin])
            std_S[mask_basin] = np.ones(len(data.temperature_prop.data[mask_basin])) * np.nanstd(data.salinity_prop.data[mask_basin])
        year_label = []
    month_label = []
    for i in range(len(std_S[masks.closed_cavities == 1])):
        year_label.append(year)
        month_label.append(month)
    df_total = pd.DataFrame({
                            'lat': data.lat.data[masks.closed_cavities == 1], 
                            'lon': data.lon.data[masks.closed_cavities == 1], 
                            'temperature_prop': data.temperature_prop.data[masks.closed_cavities == 1],
                            'salinity_prop': data.salinity_prop.data[masks.closed_cavities ==1],
                            'melt_m_ice_per_y': data.melt_ice_per_yr.data[masks.closed_cavities == 1],
                            'mean_T': mean_T[masks.closed_cavities == 1],
                            'mean_S': mean_S[masks.closed_cavities == 1],
                            'std_T': std_T[masks.closed_cavities == 1],
                            'std_S': std_S[masks.closed_cavities == 1], 
                            'year': year_label, 
                            'month': month_label, 
                            'basins_NEMO': basins_merged.data[masks.closed_cavities == 1],
                            'distances_GL': geoms.distances_GL.data[masks.closed_cavities == 1],
                            'distances_OO': geoms.distances_OO.data[masks.closed_cavities == 1],
                            'distances_OC': geoms.distances_OC.data[masks.closed_cavities == 1],
                            'corrected_isdraft': geoms.isf_draft.data[masks.closed_cavities ==1],
                            'area': geoms.areas.data[masks.closed_cavities == 1],
                            'bathymetry': geoms.bathymetry.data[masks.closed_cavities == 1],
                            'slope_is_lon': geoms.slope_isdraft_lon.data[masks.closed_cavities == 1],
                            'slope_is_lat': geoms.slope_isdraft_lat.data[masks.closed_cavities == 1],
                            'slope_ba_lon': geoms.slope_bathy_lon.data[masks.closed_cavities == 1],
                            'slope_ba_lat': geoms.slope_bathy_lat.data[masks.closed_cavities == 1],
                            'slope_is_across_front': geoms.slope_is_across_front.data[masks.closed_cavities == 1],
                            'slope_is_towards_front': geoms.slope_is_towards_front.data[masks.closed_cavities == 1],
                            'slope_ba_across_front': geoms.slope_ba_across_front.data[masks.closed_cavities == 1],
                            'slope_ba_towards_front': geoms.slope_ba_towards_front.data[masks.closed_cavities == 1],
                            })
    if np.sum(np.isnan(df_total['temperature_prop'])) != 0:
        print(np.sum(np.isnan(df_total['temperature_prop'])) == 0, 'nan values were removed')
    df_total2 = df_total[~np.isnan(df_total['temperature_prop'])]
    return df_total2

intermediate_filepath = data_out_fp + 'Christoph_v2_' + 'whole_dataset' + '_' + 'not_yet_normalised.csv'

# Use the function create_df_total to create pandas dataframes for each simulation
# And then merge these together
# If you already have an merged but not normalised dataframe, then you can skip this step

merge = True
if merge == True:
    years_to_merge = unique_years_int
    months_to_merge = unique_months
    for i,j in itertools.product(range(len(years_to_merge)), range(len(months_to_merge))):
        if i + j < 1:
            year = str(int(years_to_merge[0]))
            month = months_to_merge[0]
            df_total = create_df_total(year, month,  masks, basins_merged, geoms)
        else:
            year = str(int(years_to_merge[i]))
            month = months_to_merge[j]
            df_ij = create_df_total(year, month,  masks, basins_merged, geoms)
            df_total2 = pd.concat([df_total, df_ij], ignore_index=True)
            df_total = df_total2
        print((i * len(months_to_merge)) + j + 1, 'out of', (len(years_to_merge)* len(months_to_merge)), 'processed', end = '\r')   

    intermediate_save = True
    if intermediate_save == True:
        df_total.to_csv(intermediate_filepath, index = False)
        print('You have saved:        ')
        print(intermediate_filepath)


