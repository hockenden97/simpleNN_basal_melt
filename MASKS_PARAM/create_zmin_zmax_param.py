import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

dsDom = xr.open_dataset('domain_cfg_eANT025.L121.nc',drop_variables={'x','y'})
dsRef = xr.open_dataset('mesh_mask_eANT025.L121_REF.nc',drop_variables={'x','y'})
dsReduced = xr.open_dataset('mesh_mask_eANT025.L121_RED_CAV.nc',drop_variables={'x','y'})

zdraft = dsDom.isf_draft.squeeze()
zbathy = dsDom.bathy_metry.squeeze()
mt,my,mx = dsDom.glamt.shape

isf_Red = dsReduced.misf.where(dsReduced.misf>1,0,drop=False).squeeze()
isf_Red = isf_Red.where(isf_Red==0,1,drop=False)
grd_Red = 1 - dsReduced.tmaskutil.squeeze()

isf_Ref = dsRef.misf.where(dsRef.misf>1,0,drop=False).squeeze()
isf_Ref = isf_Ref.where(isf_Ref==0,1,drop=False)
grd_Ref = 1 - dsRef.tmaskutil.squeeze()
oce_Ref = 1 - isf_Ref - grd_Ref

# ice shelf points in REF but not in RED_CAV:
diff = isf_Ref-isf_Red

# Sub-ice-shelf ocean points corresponding to the grounding zone in RED_CAV:
mskGZ = isf_Red * ( diff.roll(x=1) + diff.roll(x=-1) + diff.roll(y=1) + diff.roll(y=-1) )
mskGZ = mskGZ.where(mskGZ==0,1,drop=False)

# Open ocean points corresponding to the ice shelf front of cavities not resolved in RED_CAV:
mskIF = oce_Ref * ( diff.roll(x=1) + diff.roll(x=-1) + diff.roll(y=1) + diff.roll(y=-1) )
mskIF = mskIF.where(mskIF==0,1,drop=False)

# Average nearby ice-draft and local bathymetry :
zmax = (mskIF+mskGZ) * zbathy
zmin = (mskIF+mskGZ) * ( (zdraft*diff).roll(x=1) + (zdraft*diff).roll(x=-1) + (zdraft*diff).roll(y=1) + (zdraft*diff).roll(y=-1) ) / ( diff.roll(x=1) + diff.roll(x=-1) + diff.roll(y=1) + diff.roll(y=-1) + 1.e-12 )

#plt.pcolormesh(zmax)
#plt.colorbar()
#plt.show()

# save to netcdf:
ds = xr.Dataset(
    {
    "zmin":    (["y","x"], np.float32(zmin.values)),
    "zmax":    (["y","x"], np.float32(zmax.values)),
    },
    coords={
    "x": np.float64(np.arange(1,mx+1)),
    "y": np.float64(np.arange(1,my+1)),
    },
)

ds.attrs['history'] = 'created using create_zmin_zmax_param.py'

ds.to_netcdf('param_zmin_zmax_eANT025.L121_RED_CAV.nc')
