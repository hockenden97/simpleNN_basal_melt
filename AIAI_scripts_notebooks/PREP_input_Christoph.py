import glob
import os
import numpy as np
import xarray as xr
import sys 
import time

where_are_you_running = 'bettik'#, 'bettik'
# Set the filepaths for the data 
if where_are_you_running == 'bettik':
    filepath_base = '/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/'
elif where_are_you_running == 'laptop':
    filepath_base = '/Users/ockendeh/Documents/simpleNN_basal_melt/'
else:
    print('Help: I don\'t know where to look for files')

# Set all the filepaths that will be required 
# For loading in the cavity geometry and creating the T/S profiles mask 
# Domain file from Christoph
filepath_cfg = filepath_base + 'AIAI_data/Christoph/domain_cfg_eANT025.L121_notime.nc'
# Or from Nico?
#filepath_cfg = '../MASKS_PARAM/domain_cfg_eANT025.L121.nc'
# Domain file for the reduced cavity geometry 
filepath_cfg_reduced_cav = filepath_base + 'MASKS_PARAM/domain_cfg_eANT025.L121_REDUCED_CAV.nc'
# Basin numbers 
filepath_basin_nos = filepath_base + 'MASKS_PARAM/basin_raster_extrapolate_eANT025.nc'

# For calculating the T/S profiles, you need the NEMO output data 
filepath_Christoph = filepath_base + 'AIAI_data/Christoph/'

# The output nc files include
# A geometry file with the masks of the different types of cell and the basins 
filepath_mask = filepath_base +  'AIAI_data/Christoph/geometric_masks.nc'
# A geometry file with the required variables for the NN 
filepath_geomvars = filepath_base + 'AIAI_data/Christoph/geom_vars.nc'

# The data files to be processed 
filepaths_so = glob.glob(filepath_Christoph + 'so-eANT025.L121-isf94-1m_*.nc')
filepaths_fwfisf = glob.glob(filepath_Christoph + 'fwfisf-eANT025.L121-isf94-1m_*.nc')
filepaths_thetao = glob.glob(filepath_Christoph + 'thetao-eANT025.L121-isf94-1m_*.nc')

# The output file path for the T,S, melt files 
filepaths_TSmelt = filepath_Christoph + 'TSmelt_v2'

def load_geom_files(filepath_geomvars, filepath_mask, join_ice_shelves = False):
    geoms = xr.open_dataset(filepath_geomvars)
    masks = xr.open_dataset(filepath_mask)
    print('You have loaded:', filepath_geomvars)
    print('You have laoded:', filepath_mask)
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

def load_TSmelt_files(filepaths_so, filepaths_thetao, filepaths_fwfisf, year):
    for i in range(len(filepaths_so)):
            if filepaths_so[i].split(os.sep)[-1].split('.nc')[0].split('_')[-1] == str(year):
                # Import the chosen file
                so = xr.open_dataset(filepaths_so[i])
                print('You have loaded:', filepaths_so[i])
            if filepaths_thetao[i].split(os.sep)[-1].split('.nc')[0].split('_')[-1] == str(year):
                thetao = xr.open_dataset(filepaths_thetao[i])
                print('You have loaded:', filepaths_thetao[i])
            if filepaths_fwfisf[i].split(os.sep)[-1].split('.nc')[0].split('_')[-1] == str(year):
                fwfisf = xr.open_dataset(filepaths_fwfisf[i])
                print('You have loaded:', filepaths_fwfisf[i])
    return so, thetao, fwfisf
    
def extractTSmelt(masks, basins_merged, geoms, so, thetao, fwfisf, filepaths_TSmelt, year, month, delib = False):
    # To get the x,y,z, grid for a particular time slice 
    so_time = so.drop_vars('time_centered').so.isel(time_counter = month).drop_vars('time_counter')
    thetao_time = thetao.drop_vars('time_centered').thetao.isel(time_counter = month).drop_vars('time_counter')
    fwfisf_time = fwfisf.drop_vars('time_centered').fwfisf.isel(time_counter = month).drop_vars('time_counter')
    # Note changed drop to drop_vars to work better with this version of xarray
    # Then need to loop over each basin to extract the temperature profiles in the correct region 
    # Create arrays for the profiles to be put into 
    profiles_T = xr.where(so_time == 0,0,0)
    profiles_S = xr.where(so_time == 0,0,0)
    for jj in range(len(np.unique(masks.basins_NEMO))):
        print(jj+1, 'out of', len(np.unique(masks.basins_NEMO)), end = '\r')
        basin_no = jj+1
        mask_TS_basin = xr.where((basins_merged == basin_no) & (masks.mask_TS_profiles == 1), 1, np.nan) == 1
        if np.sum(mask_TS_basin) != 0:
            if delib == False:
                extract = thetao_time.where(mask_TS_basin == 1).stack(all_dims=['y_grid_T', 'x_grid_T']).dropna('all_dims')
                extract_nan = xr.where(extract == 0,np.nan, extract)
                mean_profile = extract_nan.mean(dim = 'all_dims', skipna = True)
            elif delib == True:
                thetao_nan = xr.where(thetao_time == 0, np.nan, thetao_time)
                mean_profile = thetao_nan.where(mask_TS_basin == 1).mean(['y_grid_T', 'x_grid_T'], skipna = 'True')
            filled_profileT = mean_profile.interpolate_na(dim = 'deptht').ffill(dim = 'deptht').bfill(dim = 'deptht')
            if delib == False:
                extract = so_time.where(mask_TS_basin == 1).stack(all_dims=['y_grid_T', 'x_grid_T']).dropna('all_dims')
                extract_nan = xr.where(extract == 0,np.nan, extract)
                mean_profile = extract_nan.mean(dim = 'all_dims', skipna = True)
            elif delib == True:
                so_nan = xr.where(so_time == 0, np.nan, so_time)
                mean_profile = so_nan.where(mask_TS_basin == 1).mean(['y_grid_T', 'x_grid_T'], skipna = 'True')
            filled_profileS = mean_profile.interpolate_na(dim = 'deptht').ffill(dim = 'deptht').bfill(dim = 'deptht')
            # Expand xarrrays to add them all to an x,y,z grid which the depth mask can then be applied to 
            profileT_expanded = filled_profileT.expand_dims({"y_grid_T": so_time.y_grid_T, "x_grid_T": so_time.x_grid_T})
            profileT_basin = profileT_expanded*(basins_merged == basin_no)
            profiles_T = profiles_T + profileT_basin
            profileS_expanded = filled_profileS.expand_dims({"y_grid_T": so_time.y_grid_T, "x_grid_T": so_time.x_grid_T})
            profileS_basin = profileS_expanded*(basins_merged == basin_no)
            profiles_S = profiles_S + profileS_basin
    draft = geoms.isf_draft.rename({'x': 'x_grid_T', 'y':'y_grid_T'})#*masks.closed_cavities_nan
    interp_S = profiles_S.drop_vars(['nav_lat_grid_T','nav_lon_grid_T','y_grid_T', \
                          'x_grid_T']).sel(deptht = draft, method = 'nearest')*masks.closed_cavities_nan
    interp_T = profiles_T.drop_vars(['nav_lat_grid_T','nav_lon_grid_T','y_grid_T', \
                          'x_grid_T']).sel(deptht = draft, method = 'nearest')*masks.closed_cavities_nan
    print(np.sum(np.isnan(interp_T.data[masks.closed_cavities == 1])), 'nans in temperature')
    print(np.sum(np.isnan(interp_S.data[masks.closed_cavities == 1])), 'nans in salinity')
    ds = xr.Dataset(
                data_vars=dict(
                    temperature_prop    = (["y", "x"], interp_S.data),
                    salinity_prop       = (["y", "x"], interp_T.data),
                    melt_ice_per_yr     = (["y", "x"], (fwfisf_time.values*masks.closed_cavities_nan).data),
                    ),
                coords=dict(
                    lon=(("y", "x"), interp_S.lon.data),
                    lat=(("y", "x"), interp_S.lat.data),    ),
                attrs=dict(description = "Pointwise T, S, and melt", 
                           simulation_run = 'Christoph_CFG',
                           year = year,
                           month = month)
                    )
    if delib == False:
        fp_TSmelt = filepaths_TSmelt + '_y' + str(year) + '_m' + str(month).zfill(2) +'.nc'
    elif delib == True:
        fp_TSmelt = filepaths_TSmelt + '_nonanmean' + '_y' + str(year) + '_m' + str(month).zfill(2) +'.nc'
    ds.to_netcdf(fp_TSmelt)
    print('You have saved the following file')
    print(fp_TSmelt)

def extractTSmelt_v2(masks, basins_merged, geoms, so, thetao, fwfisf, filepaths_TSmelt, year, month):
    # To get the x,y,z, grid for a particular time slice 
    so_time = so.drop_vars('time_centered').so.isel(time_counter = month).drop_vars('time_counter')
    thetao_time = thetao.drop_vars('time_centered').thetao.isel(time_counter = month).drop_vars('time_counter')
    fwfisf_time = fwfisf.drop_vars('time_centered').fwfisf.isel(time_counter = month).drop_vars('time_counter')
    thetao_time = xr.where(thetao_time == 0, np.nan, thetao_time)
    so_time = xr.where(so_time == 0, np.nan, so_time)
    # Calculate the basin numbers of any basin which contains closed cavities 
    basin_nos_temp = np.unique(basins_merged)
    count = np.zeros(len(basin_nos_temp))
    for i in range(len(basin_nos_temp)):
        count[i] = np.sum((basins_merged*masks.closed_cavities_nan) == basin_nos_temp[i])
    mask_keep_nos = count != 0
    basin_nos = basin_nos_temp[mask_keep_nos]
    # Create empty arrays for the propagated temperature and salinity (on NEMO grid)
    T_prop = np.ones(so.nav_lat_grid_T.data.shape)*np.nan
    S_prop = np.ones(so.nav_lat_grid_T.data.shape)*np.nan
    for kk in range(len(basin_nos)):
    #kk = 55
        basin_no = basin_nos[kk]
        mask_TS_basin = xr.where((basins_merged == basin_no) & (masks.mask_TS_profiles == 1), 1, np.nan) == 1
        real_profile = thetao_time.where(mask_TS_basin == 1).mean(dim = ['x_grid_T','y_grid_T'], skipna = True)
        T_profile = real_profile.interpolate_na(dim = 'deptht').ffill(dim = 'deptht').bfill(dim = 'deptht')
        real_profile = so_time.where(mask_TS_basin == 1).mean(dim = ['x_grid_T','y_grid_T'], skipna = True)
        S_profile = real_profile.interpolate_na(dim = 'deptht').ffill(dim = 'deptht').bfill(dim = 'deptht')
        mask_basin = basins_merged*masks.closed_cavities_nan == basin_no
        T_prop[mask_basin] = T_profile.sel(deptht = geoms.isf_draft.data[mask_basin], method = 'nearest')
        S_prop[mask_basin] = S_profile.sel(deptht = geoms.isf_draft.data[mask_basin], method = 'nearest')
        #print('Currently processing:', year, month, ',', \
        #          kk+1, 'out of', len(basin_nos), 'profiles processed', end = '\r')
    ds = xr.Dataset(
                data_vars=dict(
                    temperature_prop    = (["y", "x"], T_prop.data),
                    salinity_prop       = (["y", "x"], S_prop.data),
                    melt_ice_per_yr     = (["y", "x"], (fwfisf_time.values*masks.closed_cavities_nan).data),
                    ),
                coords=dict(
                    lon=(("y", "x"), so.nav_lon_grid_T.data),
                    lat=(("y", "x"), so.nav_lat_grid_T.data),    ),
                attrs=dict(description = "Pointwise T, S, and melt", 
                           simulation_run = 'Christoph_CFG',
                           year = year,
                           month = month)
                    )
    fp_TSmelt = filepaths_TSmelt + '_y' + str(year) + '_m' + str(month).zfill(2) +'.nc'
    ds.to_netcdf(fp_TSmelt)
    print('You have saved the following file')
    print(fp_TSmelt)

# Load the non-specific geometry files
masks, basins_merged, geoms = load_geom_files(filepath_geomvars, filepath_mask, join_ice_shelves = True)

years_here = []
for i in range(len(filepaths_so)):
    year_here = filepaths_so[i].split(os.sep)[-1].split('.nc')[0].split('_')[-1]
    years_here.append(int(year_here))
years_unique = np.unique(years_here)
months_unique = np.arange(0,12,1)
years_unique, months_unique 

# Load in the T, S, melt files for the specified year 
year = sys.argv[1]
print(year)
so, thetao, fwfisf = load_TSmelt_files(filepaths_so, filepaths_thetao, filepaths_fwfisf, year)

verify_if_run = False
if verify_if_run == True:
    # Verify which years have not yet run
    years = []
    for i in range(len(filepaths_thetao)):
        years.append(filepaths_thetao[i].split('_')[-1].split('.nc')[0])
    months = np.arange(0,12,1)
    years, months = np.meshgrid(years, months)
    years, months = np.ndarray.flatten(years), np.ndarray.flatten(months)
    filepaths_have_run = glob.glob(filepaths_TSmelt + '*.nc')
    years_have_run = []
    months_have_run = []
    for i in range(len(filepaths_have_run)):
        years_have_run.append(filepaths_have_run[i].split('/')[-1].split('_')[1].split('y')[1])
        months_have_run.append(int(filepaths_have_run[i].split('/')[-1].split('_')[2].split('.nc')[0].split('m')[1]))
    for i in range(12):
        if (str(year), i) in list(zip(years_have_run, months_have_run)):
            a = 0
        else:
            print(i, 'did not run before, reprocessing.')
            extractTSmelt_v2(masks, basins_merged, geoms, so, thetao, fwfisf, filepaths_TSmelt, year, i)
else:
    for i in range(12):
        start_time = time.time()
        extractTSmelt_v2(masks, basins_merged, geoms, so, thetao, fwfisf, filepaths_TSmelt, year, i)
        end_time = time.time()
        print('Processing time: {:.0f} seconds'.format(end_time - start_time))
    #print('Redoing 10')
    #extractTSmelt(masks, basins_merged, geoms, so, thetao, fwfisf, filepaths_TSmelt, year, 11)







