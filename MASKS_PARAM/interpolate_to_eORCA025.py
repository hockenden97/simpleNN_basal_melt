import numpy as np
import xarray as xr
from scipy import interpolate

# basin and coordinate of ISMIP6 grid :
dsmsk = xr.open_dataset('basin_raster_extrapolate.nc')
dsll  = xr.open_dataset('lon_lat_8km.nc')

my,mx=dsll.lon.shape
print('mx,my=',mx,my)

basin_1d = np.reshape(dsmsk.basins.values,mx*my)
lon_1d   = np.reshape(dsll.lon.values,mx*my)
lat_1d   = np.reshape(dsll.lat.values,mx*my)

# NEMO grid:
dsORCA = xr.open_dataset('/ccc/workflash/cont003/gen6035/jourdain/input/nemo_eANT025.L121/mesh_mask_eANT025.L121.nc')
mtNEMO,myNEMO,mxNEMO = dsORCA.glamt.shape
print('mxNEMO,myNEMO,mtNEMO =',mxNEMO,myNEMO,mtNEMO)
lon_out_1d = np.reshape(dsORCA.glamt.values,mxNEMO*myNEMO)
lat_out_1d = np.reshape(dsORCA.gphit.values,mxNEMO*myNEMO)

# interpolation:
basin_NEMO1d = interpolate.griddata( (lon_1d,lat_1d), basin_1d, (lon_out_1d,lat_out_1d), method='nearest' )
basin_NEMO = np.reshape( basin_NEMO1d, (myNEMO,mxNEMO) )

# treatment to smooth boundaries between two regions (otherwise wriggles and isolated points appear):
print(basin_NEMO)
for Nbasin in np.arange(381):
    print('Basin: ',Nbasin)
    msk=np.where(abs(basin_NEMO-Nbasin)<0.1,1,0)
    print(np.sum(msk))
    # first point of extension
    ngh =  np.roll(msk,( 0, 1),axis=(0,1)) + np.roll(msk,( 1, 1),axis=(0,1)) \
         + np.roll(msk,( 1, 0),axis=(1,0)) + np.roll(msk,( 1,-1),axis=(0,1)) \
         + np.roll(msk,( 0,-1),axis=(0,1)) + np.roll(msk,(-1,-1),axis=(0,1)) \
         + np.roll(msk,(-1, 0),axis=(0,1)) + np.roll(msk,(-1, 1),axis=(0,1)) + msk
    new_msk=msk
    new_msk[ngh>1]=1 # at least two points around to extend
    # second point of extension :
    msk=new_msk
    ngh =  np.roll(msk,( 0, 1),axis=(0,1)) + np.roll(msk,( 1, 1),axis=(0,1)) \
         + np.roll(msk,( 1, 0),axis=(1,0)) + np.roll(msk,( 1,-1),axis=(0,1)) \
         + np.roll(msk,( 0,-1),axis=(0,1)) + np.roll(msk,(-1,-1),axis=(0,1)) \
         + np.roll(msk,(-1, 0),axis=(0,1)) + np.roll(msk,(-1, 1),axis=(0,1)) + msk
    new_msk=msk
    new_msk[ngh>1]=1 # at least two points around to extend
    # 1st point of contraction:
    ngh =  np.roll(new_msk,( 0, 1),axis=(0,1)) + np.roll(new_msk,( 1, 1),axis=(0,1)) \
         + np.roll(new_msk,( 1, 0),axis=(1,0)) + np.roll(new_msk,( 1,-1),axis=(0,1)) \
         + np.roll(new_msk,( 0,-1),axis=(0,1)) + np.roll(new_msk,(-1,-1),axis=(0,1)) \
         + np.roll(new_msk,(-1, 0),axis=(0,1)) + np.roll(new_msk,(-1, 1),axis=(0,1)) + new_msk
    msk=new_msk
    msk[ngh<9]=0 # contract if 1 neighbour with new_mask=0
    # 2nd point of contraction:
    new_msk=msk
    ngh =  np.roll(new_msk,( 0, 1),axis=(0,1)) + np.roll(new_msk,( 1, 1),axis=(0,1)) \
         + np.roll(new_msk,( 1, 0),axis=(1,0)) + np.roll(new_msk,( 1,-1),axis=(0,1)) \
         + np.roll(new_msk,( 0,-1),axis=(0,1)) + np.roll(new_msk,(-1,-1),axis=(0,1)) \
         + np.roll(new_msk,(-1, 0),axis=(0,1)) + np.roll(new_msk,(-1, 1),axis=(0,1)) + new_msk
    msk=new_msk
    msk[ngh<9]=0 # contract if 1 neighbour with new_mask=0
    print(np.sum(msk),np.sum((msk==1)))
    basin_NEMO[msk==1]=Nbasin

# merge basins:
basin_NEMO = np.where(abs(basin_NEMO-34)<0.1,35,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-50)<0.1,51,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-77)<0.1,76,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-58)<0.1,76,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-143)<0.1,76,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-108)<0.1,110,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-109)<0.1,110,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-105)<0.1,105,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-140)<0.1,353,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-63)<0.1,139,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-138)<0.1,139,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-72)<0.1,152,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-183)<0.1,184,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-189)<0.1,184,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-181)<0.1,192,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-182)<0.1,192,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-185)<0.1,192,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-186)<0.1,192,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-187)<0.1,188,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-98)<0.1,153,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-179)<0.1,137,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-104)<0.1,137,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-178)<0.1,137,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-180)<0.1,137,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-99)<0.1,166,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-100)<0.1,166,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-101)<0.1,166,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-165)<0.1,166,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-102)<0.1,175,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-167)<0.1,175,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-168)<0.1,175,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-169)<0.1,175,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-170)<0.1,175,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-171)<0.1,175,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-172)<0.1,175,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-173)<0.1,175,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-174)<0.1,175,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-175)<0.1,175,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-159)<0.1,160,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-161)<0.1,97,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-0)<0.1,158,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-116)<0.1,94,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-117)<0.1,94,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-5)<0.1,9,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-15)<0.1,17,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-18)<0.1,19,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-20)<0.1,22,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-28)<0.1,193,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-120)<0.1,26,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-121)<0.1,26,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-119)<0.1,142,basin_NEMO)

# manual corrections [y,x]:
basin_NEMO[325-1:329, 212-1:215]= 43
basin_NEMO[312-1:321, 286-1:295]= 51
basin_NEMO[304-1:307, 316-1:324]= 54
basin_NEMO[285-1:286, 354-1:355]= 89
basin_NEMO[279-1:281, 364-1:366]=125
basin_NEMO[232-1:233, 372-1:375]=123
basin_NEMO[206-1:211, 361-1:374]= 56
basin_NEMO[212-1:213, 361-1:374]=122
basin_NEMO[164-1:179, 379-1:396]=901
basin_NEMO[158-1:163, 374-1:391]=902
basin_NEMO[131-1:136, 364-1:370]= 80
basin_NEMO[118-1:120,     370-1]= 87
basin_NEMO[ 45-1:46 , 418-1:412]= 83
basin_NEMO[ 30-1:48 , 470-1:472]= 79
basin_NEMO[ 50-1:86 , 500-1:510]= 73
basin_NEMO[129-1:144, 522-1:538]=903
basin_NEMO[130-1:170, 451-1:497]=904
basin_NEMO[130-1:150, 498-1:503]=904
basin_NEMO[130-1:140, 504-1:521]=904
basin_NEMO[149-1:170, 498-1:505]=905
basin_NEMO[141-1:160, 506-1:517]=905
basin_NEMO[165-1:176, 506-1:520]=111
basin_NEMO[175-1:195, 540-1:550]=144
basin_NEMO[210-1:216, 580-1:585]= 61
basin_NEMO[215-1:222, 597-1:601]=364
basin_NEMO[220-1:228, 609-1:638]=906
basin_NEMO[229-1:235, 609-1:636]=906
basin_NEMO[224-1:245, 676-1:690]=907
basin_NEMO[227-1:228, 743-1:745]=148
basin_NEMO[240-1:244, 739-1:746]= 67
basin_NEMO[244-1:259, 732-1:752]=908
basin_NEMO[260-1:270, 737-1:755]=909
basin_NEMO[256-1:270, 756-1:767]=910
basin_NEMO[240-1:260, 784-1:793]=911
basin_NEMO[248-1:253, 797-1:804]=151
basin_NEMO[240-1:246, 850-1:870]=153
basin_NEMO[275-1:293, 890-1:940]=912
basin_NEMO[247-1:274, 905-1:950]=162
basin_NEMO[248-1:274, 903-1:904]=162
basin_NEMO[249-1:274, 900-1:902]=162
basin_NEMO[250-1:274, 893-1:899]=162
basin_NEMO[252-1:274, 893-1:899]=162
basin_NEMO[119-1:172, 900-1:923]=913
basin_NEMO[119-1:150, 877-1:899]=914
basin_NEMO[125-1:150, 865-1:876]=915
basin_NEMO[151-1:172, 870-1:899]=915
basin_NEMO[107-1:128, 930-1:965]=917
basin_NEMO[115-1:210, 966-1:981]=917
basin_NEMO[120-1:210, 930-1:940]=918
basin_NEMO[129-1:210, 941-1:965]=918
basin_NEMO[223-1:245, 1059-1:1071]=919
basin_NEMO[255-1:268, 1080-1:1103]=4
basin_NEMO[318-1:326, 1375-1:1379]=193
basin_NEMO[300-1:310, 1429-1:1435]=26
basin_NEMO[295-1:310, 1435-1:1440]=920
basin_NEMO[296-1:310, 1-1:2]=920
basin_NEMO[280-1:295, 1-1:6]=142
basin_NEMO[290-1:295, 7-1:12]=22

# regive regular numbers from 1 to N
new_Nbasin=0
for Nbasin in np.arange(np.amax(basin_NEMO)+1):
   tmp=np.sum(np.where(abs(basin_NEMO-Nbasin)<0.1,1,0))
   if ( tmp > 0.5 ):
       new_Nbasin=new_Nbasin+1
       tmp = np.where(abs(basin_NEMO-Nbasin)<0.1,new_Nbasin,basin_NEMO)
       basin_NEMO = tmp

# further corrections:
basin_NEMO[290-1:295, 7-1:12]=22
basin_NEMO = np.where(abs(basin_NEMO-29)<0.1,30,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-95)<0.1,99,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-100)<0.1,102,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-73)<0.1,74,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-62)<0.1,63,basin_NEMO)
basin_NEMO[145-1:150, 501-1:507]=151
basin_NEMO[140-1:146, 515-1:522]=151
basin_NEMO[139-1:144, 503-1:509]=150
basin_NEMO = np.where(abs(basin_NEMO-86)<0.1,84,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-50)<0.1,87,basin_NEMO)
basin_NEMO[153-1:161, 513-1:520]=88
basin_NEMO[114-1:118, 886-1:923]=300
basin_NEMO[109-1:113, 893-1:922]=300
basin_NEMO[106-1:108, 900-1:919]=300
basin_NEMO = np.where(abs(basin_NEMO-18)<0.1,19,basin_NEMO)
new_Nbasin=0
for Nbasin in np.arange(np.amax(basin_NEMO)+1):
   tmp=np.sum(np.where(abs(basin_NEMO-Nbasin)<0.1,1,0))
   if ( tmp > 0.5 ):
       new_Nbasin=new_Nbasin+1
       tmp = np.where(abs(basin_NEMO-Nbasin)<0.1,new_Nbasin,basin_NEMO)
       basin_NEMO = tmp
# again...
basin_NEMO = np.where(abs(basin_NEMO-66)<0.1,67,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-124)<0.1,114,basin_NEMO)
basin_NEMO = np.where(abs(basin_NEMO-125)<0.1,114,basin_NEMO)
new_Nbasin=0
for Nbasin in np.arange(np.amax(basin_NEMO)+1):
   tmp=np.sum(np.where(abs(basin_NEMO-Nbasin)<0.1,1,0))
   if ( tmp > 0.5 ):
       new_Nbasin=new_Nbasin+1
       tmp = np.where(abs(basin_NEMO-Nbasin)<0.1,new_Nbasin,basin_NEMO)
       basin_NEMO = tmp

# save to netcdf:
ds = xr.Dataset(
    {
    "basin":    (["y","x"], np.float32(basin_NEMO)),
    },
    coords={
    "x": np.float64(np.arange(1,mxNEMO+1)),
    "y": np.float64(np.arange(1,myNEMO+1)),
    },
)

ds.attrs['history'] = 'extrapolated using interpolate_to_eORCA025.py'
ds.attrs['max_basin_nunber'] = np.int32(Nbasin)

ds.to_netcdf('basin_raster_extrapolate_eANT025.nc')
