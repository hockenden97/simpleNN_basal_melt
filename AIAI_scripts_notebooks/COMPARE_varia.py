import numpy as np
import xarray as xr    
import glob
import pandas as pd
import itertools
import sklearn

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import sys 

data_out_fp = '/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_data/Training_data/'

this_collection = 'OPM026_OPM0263_OPM031_Christoph'
collections = this_collection.split('_')
print(collections)
filepaths = []
for i in range(len(collections)):
    if collections[i] == 'Christoph':
        collections[i] = 'Christoph_v2'
    filepaths.append(data_out_fp + collections[i] + '_' + 'whole_dataset' + '_' + 'not_yet_normalised.csv')

df_0 = pd.read_csv(filepaths[0])
print('Loaded 0')
df_1 = pd.read_csv(filepaths[1])
print('Loaded 1')
df_2 = pd.read_csv(filepaths[2])
print('Loaded 2')
df_3 = pd.read_csv(filepaths[3])
print('Loaded 3')

regions = [[2,153,3,6,7,4],\
           [8,9,12,11,14,13,15],\
           [16,17,18,19,128,22],\
           [103,154],\
           [21,23,24,25,26,27,28],\
           [29,30,31,32,33,34,35,36,37,38],\
           [40,41,42,43,44],\
           [45,97,94,95,69,70,89,88,92,93,91,90,87,46,47],\
           [136,137,62,68,67,66,65,64,63,61,60,59],\
           [57,58,71,72,73,138,140,139,81,80,79,78,104],\
           [105,49,132,141,108,142],\
           [129,101,51,50,107,130,52,143],\
           [144,145,146,110,55,56,133,134],\
           [111,125,124,127,126,100],\
           [113,122,115,116],\
           [147,119,117,77,118],\
           [120,76,75,74,86,99,98,85,150,149,148,155],\
           [1,84,121,83,82,135,152,151]]

variable = sys.argv[1]
bounds = np.min((np.min(df_0[variable]), np.min(df_1[variable]), np.min(df_2[variable]), np.min(df_3[variable]))), \
np.max((np.max(df_0[variable]), np.max(df_1[variable]), np.max(df_2[variable]), np.max(df_3[variable])))
print(bounds)
bins = np.linspace(bounds[0], bounds[1],50) # isdraft

latlonbasin = df_1.groupby(['lat','lon', 'basins_NEMO'], as_index = False).mean()
latlonbasin_m = latlonbasin.groupby('basins_NEMO', as_index = False).mean()
latlonbasin_m = latlonbasin_m[['lat','lon','basins_NEMO']]


proj=ccrs.SouthPolarStereo(central_longitude=0.0)
trans=ccrs.PlateCarree()

colors = ['deeppink', 'tomato', 'firebrick', 'darkorange']

freq0s = []
freq1s = []
freq2s = []
freq3s = []
for i in range(18):
    region = i
    freq0, _ = np.histogram(df_0[df_0.basins_NEMO.isin(regions[region])][variable], bins = bins);
    freq1, _ = np.histogram(df_1[df_1.basins_NEMO.isin(regions[region])][variable], bins = bins);
    freq2, _ = np.histogram(df_2[df_2.basins_NEMO.isin(regions[region])][variable], bins = bins);
    freq3, _ = np.histogram(df_3[df_3.basins_NEMO.isin(regions[region])][variable], bins = bins);
    freq0s.append(freq0)
    freq1s.append(freq1)
    freq2s.append(freq2)
    freq3s.append(freq3)
    print(i+1, 'out of', 18, 'processed', end = '\r')

ax = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
fig = plt.figure(constrained_layout=True, figsize = (20,18))
widths = [8,8]
heights = [1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5]
gs = fig.add_gridspec(11,2, width_ratios = widths, height_ratios = heights)

titles = ['K-A', 'A-Ap', 'Ap-B', 'B-C', 'C-Cp',\
          'Cp-D','D-Dp','Dp-E', 'E-Ep', 'Ep-F', \
          'F-G', 'G-H', 'H-Hp', 'Hp-I', 'I-Ipp', \
          'Ipp-J', 'J-Jpp', 'Jpp-K']
coords = [(0.35,0.9), (0.5,0.9), (0.72,0.87), (0.8,0.68), (0.86,0.5),\
          (0.78,0.21), (0.69,0.08), (0.56,0.08), (0.53,0.31), (0.32,0.3), (0.2,0.21),\
          (0.22,0.4), (0.08,0.45), (0.07,0.59), (0.07,0.78), (0.17,0.72),\
          (0.3,0.49), (0.4,0.60)]

for i in range(11):
    ax[i] = fig.add_subplot(gs[i,0])
for i in range(7):
    ax[i+11] = fig.add_subplot(gs[i+4,1])
ax[18] = fig.add_subplot(gs[0:4,1], projection = proj)

for i in range(18):
    region = i
    ax[i].set_yticklabels([])     # Remove tick labels
    ax[i].set_yticks([]) 
    if (i != 10):
        if (i != 17):
            ax[i].set_xticks([]) 
    if i in (10,17):
        ax[i].set_xlabel(variable, fontsize = 20)
    ax[i].set_ylabel(titles[i], fontsize = 20)
    ax[i].set_ylim(-1,1);
    ax[i].tick_params(axis ='x', labelsize = 20)

    freq0 = freq0s[i]
    ax[i].plot(bins[:-1], freq0/np.max(freq0), color = 'orange')
    ax[i].plot(bins[:-1], -freq0/np.max(freq0), color = 'orange')
    ax[i].fill_between(bins[:-1], -freq0/np.max(freq0), freq0/np.max(freq0), color = 'orange', alpha = 0.2, zorder = 0, \
                      label = 'OPM026')
    freq1 = freq1s[i]
    ax[i].plot(bins[:-1], freq1/np.max(freq1), color = 'lightgrey')
    ax[i].plot(bins[:-1], -freq1/np.max(freq1), color = 'lightgrey')
    ax[i].fill_between(bins[:-1], -freq1/np.max(freq1), freq1/np.max(freq1), color = 'lightgrey', alpha = 0.2, zorder = 0, \
                      label = 'OPM0263')
    freq2 = freq2s[i]
    ax[i].plot(bins[:-1], freq2/np.max(freq2), color = 'skyblue')
    ax[i].plot(bins[:-1], -freq2/np.max(freq2), color = 'skyblue')
    ax[i].fill_between(bins[:-1], -freq2/np.max(freq2), freq2/np.max(freq2), color = 'skyblue', alpha = 0.2, zorder = 0, \
                      label = 'OPM031')
    freq3 = freq3s[i]
    ax[i].plot(bins[:-1], freq3/np.max(freq3), color = 'royalblue')
    ax[i].plot(bins[:-1], -freq3/np.max(freq3), color = 'royalblue')
    ax[i].fill_between(bins[:-1], -freq3/np.max(freq3), freq3/np.max(freq3), color = 'royalblue', alpha = 0.2, zorder = 0, \
                      label = 'Christoph')

ax[10].legend(loc = (0,-1), fontsize = 20, ncols = 4)
#fig.legend(loc = 'lower')

for i in range(len(regions)):
    for j in range(len(regions[i])):
        basin = latlonbasin[latlonbasin.basins_NEMO == regions[i][j]]
        im = ax[18].scatter(basin.lon, basin.lat, c = colors[np.mod(i,4)], \
                   transform = trans, s= 1)    
    ax[18].annotate(titles[i], xy = (0,0), xytext = (coords[i]), xycoords = 'axes fraction', transform = trans, \
                   fontsize = 20, color = colors[np.mod(i,4)])

ax[18].axis('off');

fig.savefig('FIGURES/data_distribution_' + variable +'.png', dpi = 200, bbox_inches = 'tight')