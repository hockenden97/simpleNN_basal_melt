import numpy as np
import xarray as xr    
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

ds0=xr.open_dataset('basin_raster_extrapolate_eANT025.nc')
dsREF=xr.open_dataset('domain_cfg_eANT025.L121.nc')
dsREDCAV=xr.open_dataset('domain_cfg_eANT025.L121_REDUCED_CAV.nc')
dsREF = dsREF.squeeze('time_counter')
dsREDCAV = dsREDCAV.squeeze('time_counter')

proj=ccrs.SouthPolarStereo(central_longitude=0.0)
trans=ccrs.PlateCarree()

fig, axs = plt.subplots(nrows=1,ncols=1,
                        subplot_kw={'projection': proj},
                        figsize=(15.0,15.0))
#axs = axs.ravel()

#--------------------

# (lonmin, lonmax, latmin, latmax) :
axs.set_extent( (-180, 180, -88, -65), trans )

# Add and customize meridians/parallels
gl0 = axs.gridlines(draw_labels=True, linewidth=0.75, color='gray', alpha=0.7, linestyle='--',
                       dms=True, x_inline=False, y_inline=False )
gl0.xlocator = mticker.FixedLocator(np.arange(-180,210,30))
gl0.ylocator = mticker.FixedLocator(np.arange(-90,-50,5))
gl0.xformatter = LongitudeFormatter()
gl0.yformatter = LatitudeFormatter()
gl0.xlabel_style = {'size': 12, 'color': 'gray'}
gl0.ylabel_style = {'size': 12, 'color': 'gray'}

lon2d=dsREF.glamt.values
lat2d=dsREF.gphit.values
# to avoid hexagons in cartopy:
lon2d[lon2d>=180] = lon2d[lon2d>=180.] - 360.
delta_lon=np.abs(np.diff(lon2d))
for i, start in enumerate(np.argmax(delta_lon > 180, axis=1)):
  lon2d[i, start+1:] += 360

ocean=xr.where(((dsREF.top_level>0.5)&(dsREF.top_level<1.5)),1,np.nan)
isfREF=xr.where((dsREF.top_level>1.5),1,np.nan)
closed=xr.where(((dsREF.top_level>1.5)&(dsREDCAV.top_level<0.5)),1,np.nan)

axs.pcolormesh(lon2d,lat2d,ocean, vmin=0.5, vmax=3.0, rasterized=True, cmap='Blues', transform=trans )
axs.pcolormesh(lon2d,lat2d,isfREF, vmin=0.5, vmax=1.5, rasterized=True, cmap='Greys', transform=trans )
axs.pcolormesh(lon2d,lat2d,closed, vmin=0., vmax=2.0, rasterized=True, cmap='YlOrRd', transform=trans )

Nbasin=ds0.basin.max()
ds0.basin[0:1,:]=0.
for kbasin in np.arange(1,Nbasin+1,1):
  print(kbasin, Nbasin)
  tmp = ds0.basin.values
  tmp = np.where(np.abs(tmp-kbasin)<0.1,1,0)
  axs.contour( lon2d, lat2d, tmp, [0.5], colors='k', linewidths=0.6, transform=trans )

fig.savefig('map_masks.jpg')
fig.savefig('map_masks.pdf')
