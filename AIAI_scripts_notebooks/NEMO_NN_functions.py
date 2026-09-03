# Load the necessary packages 
import xarray as xr
import numpy as np
# Load keras from tensorflow to apply the neural network 
# To avoid problems with tensorflow, add the following lines before importing it
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
# Based on the experiment name, apply the neural network to the chosen variables
import define_params 
# To apply the neural network to the processed data frame 
import pandas as pd
import time
# To transform coordinates from lat lon to x y 
from pyproj import Transformer
# To create the ocean profile from target coordinates
from scipy.spatial import cKDTree

def normalise_variable_names(ds):
    # Define expected names, and possible options for those names 
    expected = {
        "thetao": ["thetao", "votemper"],
        "so": ["so", "vosaline"]
    }
    # Create an empty list to fill with variables to be renames
    rename_dict = {}

    for correct_name, possible_names in expected.items():
        # Check if the correct name already exists 
        if correct_name in ds.data_vars:
            continue
        # Otherwise look for alternatives
        found = None
        for alt in possible_names:
            if alt in ds.data_vars:
                found = alt
                break
        # If an alternative was found, add to renaming dictionary to rename later
        if found:
            rename_dict[found] = correct_name
        else:
            # If non of the options were present
            raise ValueError(
                f"Dataset missing required variable: '{correct_name}'. "
                f"Expected one of {possible_names}"
            )
    # Apply renaming 
    return ds.rename(rename_dict)

def load_TSmelt_normal(filepath_nemo_output):
    # Load in the simulation files 
    gridT_varofint = xr.open_dataset(filepath_nemo_output)
    print('You have loaded T and S:', filepath_nemo_output)
    # And cut out variables which are not needed
    gridT_varofint = gridT_varofint.squeeze('time_counter').sel(axis_nbounds=1).drop('time_counter').drop('time_centered')
    # Crop out most of the world and just save Antarctica 
    # Specify the coordinates needed
    min_lat = -52.2
    y_trim = np.max(np.argwhere(gridT_varofint.nav_lat[:,0].data <= min_lat))
    print('Cropped to Antarctica only')
    # Take this slice 
    #gridT2 = gridT_varofint.sel(y = slice(0,y_trim+1))[['thetao','so', 'area', 'bathy', 'isf_draft']]
    gridT2 = gridT_varofint.sel(y = slice(0,y_trim+1))
    gridT2 = normalise_variable_names(gridT2)
    gridT2 = gridT2[['thetao', 'so']]#, 'area']]
    # Close the original dataset
    so = gridT2.so
    thetao = gridT2.thetao
    #return so, thetao
    return gridT2

def load_area(filepath_nemo_output):
    # Load in the simulation files 
    gridT_varofint = xr.open_dataset(filepath_nemo_output)
    print('You have loaded:', filepath_nemo_output)
    # And cut out variables which are not needed
    gridT_varofint = gridT_varofint.squeeze('time_counter').sel(axis_nbounds=1).drop('time_counter').drop('time_centered')
    # Crop out most of the world and just save Antarctica 
    # Specify the coordinates needed
    min_lat = -52.2
    y_trim = np.max(np.argwhere(gridT_varofint.nav_lat[:,0].data <= min_lat))
    print('Cropped to Antarctica only')
    # Take this slice 
    gridT2 = gridT_varofint.sel(y = slice(0,y_trim+1))
    gridT2 = normalise_variable_names(gridT2)
    gridT2 = gridT2[['area']]
    # Close the original dataset
    areas = gridT2.area
    print('Returning area')
    return areas

def load_gridcoords(filepath_nemo_output):
    # Load in the simulation files 
    gridT_varofint = xr.open_dataset(filepath_nemo_output)
    print('You have loaded coordinates:', filepath_nemo_output)
    # And cut out variables which are not needed
    gridT_varofint = gridT_varofint.squeeze('time_counter').drop('time_counter').drop('time_centered')
    # Crop out most of the world and just save Antarctica 
    # Specify the coordinates needed
    min_lat = -52.2
    y_trim = np.max(np.argwhere(gridT_varofint.nav_lat[:,0].data <= min_lat))
    print('Cropped to Antarctica only')
    # Take this slice 
    gridT2 = gridT_varofint.sel(y = slice(0,y_trim+1))[['bounds_nav_lon', 'bounds_nav_lat']]
    # Close the original dataset
    #so = gridT2.so
    #thetao = gridT2.thetao
    #return so, thetao
    return gridT2

def load_geom_files(filepath_geomvars, filepath_mask, join_ice_shelves = False):
    geoms = xr.open_dataset(filepath_geomvars)
    masks = xr.open_dataset(filepath_mask)
    print('You have loaded:', filepath_geomvars)
    print('You have loaded:', filepath_mask)
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

# Load in a trained neural network model
def load_model_nn(exp_name, this_collection, path_model, seed_nb, \
                  mod_size = 'small', TS_opt = 'extrap', norm_method = 'std', verbose = 0, annual_f = ''):
    fp_model = path_model + 'model_nn_'+ mod_size + '_' +exp_name + '_' + this_collection + '_' + annual_f + \
                                    str(seed_nb).zfill(2) + '_' + TS_opt + '_' + norm_method + '.keras'
    model = keras.models.load_model(fp_model)
    if verbose == 1:
        print('\033[1m' + 'You have loaded a trained neural network:' + '\033[0m')
        print(fp_model)
    return model

# Load in the normalisation metrics 
def load_normalisation_metrics(this_collection, path_norm_metrics, norm_method = 'std', verbose = 0):
    fp_norm_metrics = path_norm_metrics + this_collection + '_metrics_norm.nc'
    norm_metrics_file = xr.open_dataset(fp_norm_metrics)
    if verbose > 0:
        print('\033[1m' + 'You have loaded normalisation metrics:' + '\033[0m')
        print(fp_norm_metrics)
    norm_metrics = norm_metrics_file.sel(norm_method=norm_method).drop('norm_method').to_dataframe()
    return norm_metrics

def normalise_vars(var, mean, std):
    var_norm = (var - mean) / std
    return var_norm

def denormalise_vars(var_norm, mean, std):
    var = (var_norm * std) + mean
    return var

# Apply one trained NN to the chosen data 
def apply_model(model, norm_metrics, clean_df, exp_name, verbose = 0):
    # First normalise input data 
    val_norm = normalise_vars(clean_df,
                            norm_metrics.loc['mean_vars'],
                            norm_metrics.loc['range_vars'])
    # Import the desired variables, and create a list 
    var_list = define_params.var_list_by_exp_name(exp_name)
    input_vars = list(np.array(var_list)[~(np.array(var_list) == 'melt_m_ice_per_y')])
    # Split into input and reference datasets
    x_val_norm = val_norm[input_vars]
    # Apply the model 
    #shape = x_val_norm.to_array().values.shape[1], x_val_norm.to_array().values.shape[0]
    #y_out_norm = model.predict(x_val_norm.to_array().values.reshape(shape),verbose = 0)
    y_out_norm = model.predict(x_val_norm.to_array().values.T,verbose = 0)
    y_out_norm_xr = xr.DataArray(data=y_out_norm.squeeze()).rename({'dim_0': 'index'})
    y_out_norm_xr = y_out_norm_xr.assign_coords({'index': x_val_norm.index})    
    # denormalise the output
    y_out = denormalise_vars(y_out_norm_xr, 
                             norm_metrics['melt_m_ice_per_y'].loc['mean_vars'],
                             norm_metrics['melt_m_ice_per_y'].loc['range_vars'])
    if verbose > 0:
        print('\033[1m' + 'You have applied the neural network to the chosen data:' + '\033[0m')
    return y_out

# Apply all 10 NNs (with the 10 random seeds), and look at the mean 
def apply_10nns(exp_name, this_collection, clean_df, path_model, path_norm_metrics, 
                       mod_size = 'small', 
                         TS_opt = 'extrap', 
                    norm_method = 'std', 
                        verbose = 0, 
                    annual_only = False, 
               apply_collection = None, 
                       annual_f = ''):
    # Load in the normalisation metrics 
    norm_metrics = load_normalisation_metrics(this_collection, path_norm_metrics, norm_method = 'std', verbose = 0)
    # Extract the values that you might want from the clean dataframe
    df_pred = pd.DataFrame({'lat'   : clean_df.lat, 
                            'lon'   : clean_df.lon,
                            'basin' : clean_df.basins_NEMO,
                            'area'  : clean_df.area, 
                            'fraction' : clean_df.fraction})
    seed_nb_array = np.arange(1,11,1)
    list_melt_seeds = []
    start_time = time.time()
    for i in range(len(seed_nb_array)):
        list_melt_seeds.append('melt_' + str(i+1).zfill(2))
    
        model = load_model_nn(exp_name, this_collection, path_model, seed_nb_array[i], \
                                  mod_size = mod_size, TS_opt = TS_opt, norm_method = norm_method, \
                                  verbose = verbose, annual_f = annual_f) 
        df_pred['melt_' + str(i+1).zfill(2)] = apply_model(model, norm_metrics, clean_df, exp_name, verbose = verbose)
        end_time = time.time()
        print(seed_nb_array[i], 'out of', len(seed_nb_array), 'processed. Time: {:.1f} s'.format(end_time - start_time), end = '\r')
    # Calculate the ensemble mean 
    df_pred['mean_melt'] = np.mean(df_pred[list_melt_seeds], axis = 1).values
    if verbose == 1:
        print('Maximum melt rate: {:.5f} '.format(np.min(df_pred.mean_melt.values)))
        print('Minimum melt rate: {:.5f} '.format(np.max(df_pred.mean_melt.values)))
        print('Mean melt rate: {:.5f} '.format(np.mean(df_pred.mean_melt.values)))
    # And the standard deviation 
    df_pred['std_melt'] = np.std(df_pred[list_melt_seeds], axis = 1).values
    #df_pred.loc[:,'melt_Gt'] = df_pred.mean_melt * (60*60*24*365.25)/917
    df_pred.loc[:,'melt_Gt'] = df_pred.mean_melt * df_pred.area * df_pred.fraction * 31536000 * 1/1e12
    # Save just the ensemble mean/std
    #df_save = df_pred.copy()[['lat','lon', 'basin', 'area', 'mean_melt', 'std_melt', 'melt_Gt']]
    # Save just the melt by basin/
    melt_by_basin = df_pred.groupby('basin', as_index = False).sum()[['basin','melt_Gt']]
    #lat_lon_by_basin = df_pred.groupby('basin', as_index = False).mean()[['lat','lon']]
    return melt_by_basin

def expand_to_full_grid(cropped, y_trim, gridT_varofint):
    """
    cropped: data to return to full array 
    y_trim: last y-index included in the crop
    grdiT_varofint: original full grid dataset (for coords)
    """
    full = xr.DataArray(
        np.full((gridT_varofint.nav_lat.shape), np.nan),
        dims=("y", "x"),
        coords={"nav_lat": (("y", "x"), gridT_varofint.nav_lat.data),
                "nav_lon": (("y", "x"), gridT_varofint.nav_lon.data),})
    full.loc[dict(y=slice(0, y_trim+1))] = cropped
    return full

def apply_nn_to_NEMO(filepath_nemo_output, 
                     filepath_mask, 
                     filepath_geomvars, 
                     filepath_eORCA1_bmach_masks, 
                     filepath_domain_CFG,
                     path_model, 
                     path_norm_metrics, 
                     filepath_nn_output, 
                     this_collection, 
                     exp_name, 
                     join_ice_shelves, 
                     verbose = 0, 
                     nemo_version = '5'):
    ''' This function takes a NEMO T/S output file,       '''
    ''' and applies the trained neural network to it,     '''
    ''' giving an output melt file to feed back into NEMO '''

    start_time = time.time()
    # Load in the mask and geometry 
    masks, basins_merged, geoms = load_geom_files(filepath_geomvars, 
                                                  filepath_mask, 
                                                  join_ice_shelves = join_ice_shelves)
    # Rename the dimensions of basins merged to be x and y 
    if 'y_grid_T' in basins_merged.dims:
        basins_merged = basins_merged.rename({"y_grid_T": "y", "x_grid_T": "x"}).astype(int)
    # Load in the coordinates for the grid 
    coords = load_gridcoords(filepath_nemo_output)

    # The basins mask for the NEMO 025 grid  
    basins_NEMO = masks.basins_NEMO.values
    if join_ice_shelves == True:
        print('NOTE THAT THIS DOES NOT CURRENTLY JOIN THE RIGHT ICE SHELVES') 
        # Dotson and Crosson?
        #basins_NEMO[basins_NEMO == 101] = 129
        # Abbot Ice Shelf
        basins_NEMO[basins_NEMO == 109] = 143
        # George VI
        basins_NEMO[basins_NEMO == 112] = 125
        # Lambert 
        basins_NEMO[basins_NEMO == 20] = 103
    basin_nos = np.arange(1,156, dtype = 'float32')
    basin_nos = np.unique(basins_NEMO)
    
    # Load in the mask for the ocean region 
    bmach_masks = xr.open_dataset(filepath_eORCA1_bmach_masks)
    mask_nocavs = bmach_masks.mask_nocavs.values
    bmach_masks.close()
    
    # Regions where it is acceptable to extract ocean profiles 
    iy, ix = np.where((mask_nocavs == 1))
    # coordinates of those cells
    lat_ocean = coords.nav_lat.values[iy, ix]
    lon_ocean = coords.nav_lon.values[iy, ix]
    # Transform to x and y
    transformer = Transformer.from_crs("epsg:4326","epsg:3031")
    x_ocean, y_ocean = transformer.transform(lat_ocean, lon_ocean)
    # Stack to prepare for the KDTree
    ocean_points = np.column_stack((x_ocean, y_ocean))
    # Create KDTree
    tree = cKDTree(ocean_points)
    
    # Load in the TS grid 
    gridT2 = load_TSmelt_normal(filepath_nemo_output)
    print('Loaded data. Time: {:.1f} s'.format(time.time() - start_time))

    # For each basin in basin_nos, create a mask of the ocean profiles to extract 
    profiles_all = []
    i_no_value = []
    basins_all_temp = np.zeros_like(mask_nocavs)
    for kk in range(len(basin_nos)):
        i = basin_nos[kk]
        mask_basin_cavs = (basins_NEMO == i) & (masks.closed_cavities == 1)
        # Extract these points 
        idx = np.where(mask_basin_cavs == 1)
        lat_cavs = masks.lat.values[idx]
        lon_cavs = masks.lon.values[idx]
        # Transform to x and y
        transformer = Transformer.from_crs("epsg:4326","epsg:3031")
        x_cavs, y_cavs = transformer.transform(lat_cavs, lon_cavs)
        # Stack cavity points to query KDTree
        cav_points = np.column_stack((x_cavs, y_cavs))
        dist, idx = tree.query(cav_points)
        # Same dimensions as the mask of the not cavities
        ocean_edge_basin = np.zeros_like(mask_nocavs)
        # And fill the cells found with the kd-tree
        # Should allow cells which are adjacent or as close to adjacent as possible
        if len(idx) == 1:
            for k in range(len(idx)):
                ocean_edge_basin[iy[idx[k]], ix[idx[k]]] = 1
        elif len(idx) != 0:
            idxn = np.unique(idx[dist < np.min(dist[dist != np.min(dist)]) * 2])
            for k in range(len(idxn)):
                ocean_edge_basin[iy[idxn[k]], ix[idxn[k]]] = 1
        else:
            # Calculate which basins don't have profiles
            i_no_value.append(int(i))
        ### This is only needed if you want to look at which ocean cells will recieve melt from many basins   
        basins_all_temp = basins_all_temp + ocean_edge_basin
        
        # Use the mask to extract the relevant profiles 
    
        masked_depth = gridT2.where(ocean_edge_basin == 1)
        profiles = masked_depth.stack(points=("x", "y"))
        profiles = profiles.dropna(dim="points", how="all")
        #profiles = profiles.drop_vars(["area", "bathy", "isf_draft"]) 
        # Fill the nan values in the profile
        profile_kk = profiles.mean(dim = 'points').interpolate_na(dim = 'deptht').ffill('deptht').bfill('deptht')
        # Add a coordinate for kk
        profile_kk = profile_kk.expand_dims(kk =[int(i)])
        #profile_kk = profile_kk.expand_dims(kk_m1 =[int(i)-1])
        profiles_all.append(profile_kk)
        if verbose == 1:
            print('{} out of {} profiles processed'.format(kk+1, len(basin_nos)), end = '\r')
    # Concatenate the profiles along the new dimension
    all_profiles = xr.concat(profiles_all, dim="kk")
    
    print('There are {} basins with no ocean profiles : {}'.format(len(i_no_value), i_no_value))
    print('There should be 5 basins (2 with no cavities plus 3 where we merged)')
    print('Created T/S profiles. Time: {:.1f} s'.format(time.time() - start_time))

    # Extract the relevant values of temperature and salinity 
    temp_xy = all_profiles.thetao.interp(deptht = geoms.isf_draft).sel(kk=basins_merged)
    temp_xy = temp_xy.reset_coords(drop=True)
    sals_xy = all_profiles.so.interp(deptht = geoms.isf_draft).sel(kk=basins_merged)
    sals_xy = sals_xy.reset_coords(drop=True)
    
    # Calculate mean and standard deviation by region
    # Note: this step produces an error because some basins have nan mean, but the code still runs so it's okay?
    meanT_by_region = temp_xy.groupby(basins_merged).mean()
    stdT_by_region  = temp_xy.groupby(basins_merged).std()
    meanS_by_region = sals_xy.groupby(basins_merged).mean()
    stdS_by_region  = sals_xy.groupby(basins_merged).std()
    # and send back to the original grid 
    mean_T = meanT_by_region.sel(basins_NEMO = basins_merged)
    std_T = stdT_by_region.sel(basins_NEMO = basins_merged)
    mean_S = meanS_by_region.sel(basins_NEMO = basins_merged)
    std_S = stdS_by_region.sel(basins_NEMO = basins_merged)
    # Reset the coordinates
    mean_T = mean_T.reset_coords(drop=True)
    mean_S = mean_S.reset_coords(drop=True)
    std_T = std_T.reset_coords(drop=True)
    std_S = std_S.reset_coords(drop=True)

    # Create the input dataset for the neural network to be applied to
    ds = geoms.rename({"areas": "area",
                       "isf_draft": "corrected_isdraft",
                       "slope_isdraft_lon": "slope_is_lon",
                       "slope_isdraft_lat": "slope_is_lat",
                       "slope_bathy_lon": "slope_ba_lon",
                       "slope_bathy_lat": "slope_ba_lat",})
    ds = ds.assign(temperature_prop = temp_xy,
                      mean_T           = mean_T,
                      std_T            = std_T,
                      salinity_prop    = sals_xy,
                      mean_S           = mean_S,
                      std_S            = std_S,
                      basins_NEMO      = basins_merged,)
    ds.attrs["description"] = "Neural network input variables"

    # Rename variables in closed cavity mask 
    if 'y_grid_T' in basins_merged.dims:
        closed_cavs = masks.closed_cavities_nan.rename({"y_grid_T": "y", "x_grid_T": "x"})
    else:
        closed_cavs = masks.closed_cavities_nan
    # Select only the points in the closed cavity regions
    stacked = ds.where(closed_cavs == 1).stack(points=("x", "y")).dropna("points", how = "all")
    # and convert to a pandas dataframe
    selected = stacked.to_dataframe().reset_index().drop(columns=["x", "y"])
    # Remove nan values (although still unclear why there are nan values)
    selected = selected[~np.isnan(selected.temperature_prop)]
    print('Prepared data to apply NN. Time: {:.1f} s'.format(time.time() - start_time))
    
    # Apply all 10 neural networks 
    print('Applying neural networks')
    melt_by_basin = apply_10nns(exp_name, this_collection, selected.to_xarray(), path_model, path_norm_metrics)
    print('Applied NN. Time: {:.1f} s               '.format(time.time() - start_time))
    
    areas = load_area(filepath_nemo_output)
    melt_all = np.zeros_like(mask_nocavs)
    melt_b = np.zeros_like(mask_nocavs)
    melt_t = np.zeros_like(mask_nocavs)
    CFG = xr.open_dataset(filepath_domain_CFG)
    print('Putting melt back onto the ocean grid')
    for kk in range(len(melt_by_basin.basin)):
        i = melt_by_basin.basin[kk]
        mask_basin_cavs = (basins_NEMO == i) & (masks.closed_cavities == 1)
        # Calculate the top and bottom depths to inject the melt 
        # Currently using means, but might be better to use a min/max? ideally at cells touching the ocean 
        melt_b_basin = np.mean(geoms.bathymetry.values[mask_basin_cavs == 1])
        melt_t_basin = np.mean(geoms.isf_draft.values[mask_basin_cavs == 1])
        # Extract these points 
        idx = np.where(mask_basin_cavs == 1)
        lat_cavs = masks.lat.values[idx]
        lon_cavs = masks.lon.values[idx]
        # Transform to x and y
        transformer = Transformer.from_crs("epsg:4326","epsg:3031")
        x_cavs, y_cavs = transformer.transform(lat_cavs, lon_cavs)
        # Stack cavity points to query KDTree
        cav_points = np.column_stack((x_cavs, y_cavs))
        dist, idx = tree.query(cav_points)
        # Same dimensions as the mask of the not cavities
        ocean_edge_basin = np.zeros_like(mask_nocavs)
        # And fill the cells found with the kd-tree
        # Should allow cells which are adjacent or as close to adjacent as possible
        if len(idx) == 1:
            for k in range(len(idx)):
                ocean_edge_basin[iy[idx[k]], ix[idx[k]]] = 1
        elif len(idx) != 0:
            idxn = np.unique(idx[dist < np.min(dist[dist != np.min(dist)]) * 2])
            for k in range(len(idxn)):
                ocean_edge_basin[iy[idxn[k]], ix[idxn[k]]] = 1
        # Redistribute the melt over the relevant cells
        # Number of pixels?
        #pixels = np.sum(ocean_edge_basin)
        # Total area covered 
        ocean_area = np.sum(areas * ocean_edge_basin).data
        # Melt per pixel?? 
        melt_per_pixel = melt_by_basin.melt_Gt[kk] * (1e12/(60*60*24*365.25)) /ocean_area
        ocean_melt = ocean_edge_basin * melt_per_pixel
        # Apply top and bottom over the relevant cells
        #ocean_melt_b = ocean_edge_basin * melt_b_basin
        #ocean_melt_t = ocean_edge_basin * melt_t_basin
        ocean_melt_b = ocean_edge_basin * CFG.bathy_metry.isel(t=0).values[0:110]
        ocean_melt_t = ocean_edge_basin * CFG.isf_draft.isel(t=0).values[0:110]
        # Sum up the melt over all the basins
        melt_all = melt_all + ocean_melt
        melt_b = melt_b + ocean_melt_b
        melt_t = melt_t + ocean_melt_t
        if verbose == 1:
            print('{} out of {} profiles processed'.format(kk+1, len(melt_by_basin.basin)), end = '\r')
    print('Propagated melt to ocean cells. Time: {:.1f} s'.format(time.time() - start_time))

    # Expanding the grid back to the NEMO eORCA1 grid
    # gridT_varofint = xr.open_dataset('/ccc/work/cont003/gen6035/ockendeh/NEMO/eORCA1.L75/eORCA1.L75-I/eORCA1.4.3_CavsForNN_domain_cfg.nc')[['x','y']]
    gridT_varofint = xr.open_dataset(filepath_nemo_output)[['x','y']]
    min_lat = -52.2
    y_trim = np.max(np.argwhere(gridT_varofint.nav_lat[:,0].data <= min_lat))
    full_t   = expand_to_full_grid(melt_t,    y_trim, gridT_varofint)
    full_b   = expand_to_full_grid(melt_b,    y_trim, gridT_varofint)
    full_all = expand_to_full_grid(melt_all,  y_trim, gridT_varofint)

    if nemo_version == '5':
        # In NEMO 5 the melt convention has changed
        # Melting is now positive and refreezing is now negative 
        print('New NEMO version (v{}), melt is positive'.format(nemo_version))
        full_all = full_all * -1 
    else:
        # In NEMO 4 the melt convention is the same as in Pierre's simulations
        print('Old NEMO version (v{}), melt is negative'.format(nemo_version))
        full_all = full_all 
    
    # Create a netcdf dataset for the temperature and salinity profiles 
    output_melt = xr.Dataset(data_vars=dict(
                                            melt      = (["y", "x"], full_all.data), 
                                            melt_top  = (["y", "x"], full_t.data),
                                            melt_base = (["y", "x"], full_b.data),
                                           ),
                                coords=dict(
                                            lon = (("y", "x"), full_all.nav_lon.data),
                                            lat = (("y", "x"), full_all.nav_lat.data),    
                                           ),
                                 attrs=dict(
                                            description    = "Neural network output", 
                                            history        = f"{time.ctime()}: Applied neural network"),
                            )     
    output_melt = output_melt.fillna(0)
    print('Nan values filled')

    output_melt["melt"].attrs.update({"standard_name": "Sub-shelf melt for parameterised regions",
                                      "units": "kg m-2 s-1"})
    output_melt["melt_top"].attrs.update({"standard_name": "Top level to inject melt",
                                         "units": "m"})
    output_melt["melt_base"].attrs.update({"standard_name": "Bottom level to inject melt",
                                         "units": "m"})
    output_melt["lat"].attrs.update({"standard_name": "Latitude",
                                     "units": "degrees_north"})
    output_melt["lon"].attrs.update({"standard_name": "Longitude",
                                     "units": "degrees_east"})
    
    # Export to netcdf 
    filepath_this_nn_output = filepath_nn_output #+ 'nn_output_melt' '_' + simulation + '_' + year_of_interest + '.nc'
    output_melt.to_netcdf(filepath_this_nn_output)
    print('You have saved the nn output to:', filepath_this_nn_output)
    print('Saved melt. Time: {:.1f} s'.format(time.time() - start_time))














    
