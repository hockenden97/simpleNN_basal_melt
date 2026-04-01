import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

dsDom = xr.open_dataset('domain_cfg_eANT025.L121.nc',drop_variables={'x','y','nav_lev'})
dsRef = xr.open_dataset('mesh_mask_eANT025.L121_REF.nc',drop_variables={'x','y'})
dsReduced = xr.open_dataset('mesh_mask_eANT025.L121_RED_CAV.nc',drop_variables={'x','y'})
dsBasin = xr.open_dataset('basin_raster_extrapolate_eANT025.nc',drop_variables={'x','y'})

zdraft = dsDom.isf_draft.squeeze()
zbathy = dsDom.bathy_metry.squeeze()
e3t    = dsDom.e3t_1d.squeeze() 
e3w    = dsDom.e3w_1d.squeeze() 
zsup   = dsDom.e3w_1d.squeeze().cumsum().values
zinf   = dsDom.e3w_1d.squeeze().cumsum().roll(nav_lev=1).values
zinf[0]= 0.e0
e1e2t  = (dsDom.e1t*dsDom.e2t).squeeze()
mt,mz,my,mx = dsDom.e3t_0.shape

isf_Red = dsReduced.misf.where(dsReduced.misf>1,0,drop=False).squeeze()
isf_Red = isf_Red.where(isf_Red==0,1,drop=False)
grd_Red = 1 - dsReduced.tmaskutil.squeeze()

isf_Ref = dsRef.misf.where(dsRef.misf>1,0,drop=False).squeeze()
isf_Ref = isf_Ref.where(isf_Ref==0,1,drop=False)
grd_Ref = 1 - dsRef.tmaskutil.squeeze()
oce_Ref = 1 - isf_Ref - grd_Ref

# ice shelf points in REF but not in RED_CAV:
diff = isf_Ref-isf_Red

Nbasin=dsBasin.basin.max().values.astype('short')+1
print('number of basins = ',Nbasin)

area = np.zeros((Nbasin,mz))
for kbasin in np.arange(Nbasin):
  print(kbasin)
  msk = dsBasin.basin.where(dsBasin.basin==kbasin,0)
  msk = msk.where(msk==0,1)*diff # msk=1 for the ice shelf areas that have been removed in this basin
  #print(kbasin, msk.sum().values)
  if ( msk.sum().values > 0 ):
    eee = e1e2t.where(msk==1,other=np.nan).values.reshape(mx*my)
    eee = eee[~np.isnan(eee)]
    zzz = zdraft.where(msk==1,other=np.nan).values.reshape(mx*my)
    zzz = zzz[~np.isnan(zzz)]
    for kz in np.arange(mz):
      area[kbasin,kz] = eee[np.where( ((zzz<zsup[kz])&(zzz>=zinf[kz])) )].sum()

#plt.pcolormesh(zmax)
#plt.colorbar()
#plt.show()

print(area[1,0:30])

# save to netcdf:
ds = xr.Dataset(
    {
    "area":    (["basin","nav_lev"], np.float32(area)),
    #"Nmax_basin": np.short(Nbasin)
    },
    coords={
    "basin": np.float64(np.arange(1,Nbasin+1)),
    "nav_lev": np.float64(np.arange(1,mz+1)),
    },
)

ds.attrs['history'] = 'created using create_area_param.py'
ds.attrs['max_basin_nunber'] = np.int32(Nbasin)

ds.to_netcdf('param_area_eANT025.L121_RED_CAV.nc')
