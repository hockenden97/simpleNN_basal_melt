import numpy as np
import xarray as xr    

ds0=xr.open_dataset('basin_raster_extrapolate_eANT025.nc',drop_variables={'x','y'})
#-
dsREF=xr.open_dataset('domain_cfg_eANT025.L121.nc')
dsREDCAV=xr.open_dataset('domain_cfg_eANT025.L121_REDUCED_CAV.nc')
dsREF = dsREF.squeeze('time_counter')
dsREDCAV = dsREDCAV.squeeze('time_counter')
#-
dsZminmax = xr.open_dataset('param_zmin_zmax_eANT025.L121_RED_CAV.nc',drop_variables={'x','y'})

ocean=xr.where(((dsREF.top_level>0.5)&(dsREF.top_level<1.5)),1,np.nan)
isfREF=xr.where((dsREF.top_level>1.5),1,np.nan)
closed=xr.where(((dsREF.top_level>1.5)&(dsREDCAV.top_level<0.5)),1,np.nan)

Nbasin=ds0.basin.max().values.astype('int')
print(ds0.basin.min().values.astype('int'))
print(Nbasin,' basins')

for year in np.arange(1982,2016,1):
     
  file_mlt = 'DATA_eANT025_REF/eANT025.L121-REF_1m_'+year.astype('str')+'0101_'+year.astype('str')+'1231_SBC.nc'
  print(' ')
  print(file_mlt)
  dsmlt = xr.open_dataset(file_mlt,drop_variables={'x','y'})

  check_total_melt = np.zeros((12))
  check_speci_melt = np.zeros((12))
 
  for kbasin in np.arange(Nbasin):

     # msk_closed = 1 if closed area in this basin:
     msk_closed = xr.where(((dsREF.top_level>1.5)&(dsREDCAV.top_level<0.5)&(ds0.basin>kbasin+0.5)&(ds0.basin<kbasin+1.5)),1,np.nan)
     msk_check = xr.where(((dsREF.top_level>1.5)&(ds0.basin>kbasin+0.5)&(ds0.basin<kbasin+1.5)),1,np.nan)
     #print(msk_closed.sum(dim=["x","y"],skipna=True).values,msk_check.sum(dim=["x","y"],skipna=True).values)

     # msk_exchg = 1 in the area of this basin where we redistribute the freshwater:
     msk_exchg = xr.where(((dsZminmax.zmin>2)&(ds0.basin>kbasin+0.5)&(ds0.basin<kbasin+1.5)),1.e0,0.e0)
     area_exchg = (msk_exchg * dsREF.e1t * dsREF.e2t).sum(dim=["x","y"],skipna=True).values
     vol_exchg = ( msk_exchg * dsREF.e1t * dsREF.e2t * (dsZminmax.zmax-dsZminmax.zmin) ).sum(dim=["x","y"],skipna=True).values

     # basin monthly melt in kg/s
     basin_melt = (dsmlt.fwfisf * msk_closed * dsREF.e1t * dsREF.e2t).sum(dim=["x","y"],skipna=True)
     print('basin',kbasin+1,' : ',basin_melt.values[11]*1.e-12*86400*365.25,' Gt/yt ',area_exchg*1.e-6,' km2')
     check_total_melt = check_total_melt + (dsmlt.fwfisf * msk_check  * dsREF.e1t * dsREF.e2t).sum(dim=["x","y"],skipna=True).values
     check_speci_melt = check_speci_melt + (dsmlt.fwfisf * msk_closed * dsREF.e1t * dsREF.e2t).sum(dim=["x","y"],skipna=True).values

     # specified monthly melt rate in kg/m2/s
     if ( kbasin == 0 ):
        melt_spe = basin_melt * msk_exchg * (dsZminmax.zmax-dsZminmax.zmin) / vol_exchg
     elif (area_exchg > 1.e-6 ):
        melt_spe = melt_spe + basin_melt * msk_exchg * (dsZminmax.zmax-dsZminmax.zmin) / vol_exchg

  print('#### Check total melt: ',np.mean(check_total_melt)*1.e-12*86400*365.25,' Gt/yr ')
  print('#### Check spec. melt: ',np.mean(check_speci_melt)*1.e-12*86400*365.25,' Gt/yr ')

  dsout=xr.Dataset(
      {
      "melt_spe":  (["time", "y", "x"], np.float32(melt_spe.values)),
      },
      coords={
      "x":np.float32(np.arange(melt_spe.shape[2])),
      "y": np.float32(np.arange(melt_spe.shape[1])),
      "time": np.float32(np.arange(melt_spe.shape[0]))
      },
  )

  dsout.attrs['history'] = 'created by N. Jourdain using transfer_simulated_to_specified_melt.py'

  file_out = 'isf_melt_spe_eANT025_REDCAV_'+year.astype('str')+'.nc'
  dsout.to_netcdf(file_out,unlimited_dims="time")
