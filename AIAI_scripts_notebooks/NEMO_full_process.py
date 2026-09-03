# Import the necessary functions

# Personalised functions created for this NN application
from NEMO_NN_functions import *
# To convert from lat lon, to x y
from pyproj import Transformer
# To extract just the bedmachine mask within the NEMO 1 degree grid 
from shapely.geometry import Polygon
from shapely.vectorized import contains
# To extract neighbouring cells
from scipy.ndimage import binary_dilation 
# To plot figures 
import matplotlib.pyplot as plt

# To create the ocean profile 
from scipy.spatial import cKDTree
# To set variables in the bash script 
import argparse
#import sys 

# Set the filepaths and any variables 

# NEMO simulation temperature and salinity profiles 
##'/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_data/ECM71-ico-LR-pi-01.pi_1883_1Y_grid_T.nc'
#filepath_nemo_output = sys.argv[1] # Set filepath for input data 
#simulation = sys.argv[2]           # Set from the input 
#year_of_interest = sys.argv[3]     # Set from the input 
#filepath_nn_output = sys.argv[4]   # Set filepath for the output 

# Define the arguments that will be passed to the python script 
def parse_args():
    parser = argparse.ArgumentParser(description="Variables for applying NN emulator")
    parser.add_argument("--nnin", required=True,
                        help="Path to NEMO input data")
    #parser.add_argument("--sim", required=True,
    #                    help="Simulation name")
    #parser.add_argument("--year", required=True, type=int,
    #                    help="Year of interest")
    parser.add_argument("--nnout", required=True,
                        help="Path for NN output")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    filepath_nemo_output = args.nnin
    #simulation = args.sim
    #year_of_interest = args.year
    filepath_nn_output = args.nnout

# Set the filepaths 
comp = 'dahu' 
if comp == 'TGCC':
    # The target grid mask 
    filepath_mask = '/ccc/scratch/cont003/gen6035/ockendeh/NEMO/eORCA1.L75/eORCA1.L75-I/NEMO_simulations/NEMO025_bmach_geometric_masks.nc'
    # The target grid geometry
    filepath_geomvars = '/ccc/scratch/cont003/gen6035/ockendeh/NEMO/eORCA1.L75/eORCA1.L75-I/NEMO_simulations/NEMO025_bmach_geom_vars.nc'
    # The eORCA1 bedmachine mask
    filepath_eORCA1_bmach_masks = '/ccc/scratch/cont003/gen6035/ockendeh/NEMO/eORCA1.L75/eORCA1.L75-I/NEMO_simulations/eORCA1_bmach_masks.nc'
    # Neural network filepaths 
    path_model = '/ccc/scratch/cont003/gen6035/ockendeh/NEMO/eORCA1.L75/eORCA1.L75-I/NEMO_simulations/models/'
    path_norm_metrics = '/ccc/scratch/cont003/gen6035/ockendeh/NEMO/eORCA1.L75/eORCA1.L75-I/NEMO_simulations/models/'
    filepath_domain_cfg = '/ccc/work/cont003/gen6035/ockendeh/NEMO/eORCA1.L75/eORCA1.L75-I/eORCA1.4.3_CavsForNN_domain_cfg.nc'
elif comp == 'dahu':
    # The target grid mask 
    filepath_mask = '/bettik/ockendeh/NEMO_simulations/NEMO025_bmach_geometric_masks.nc'
    # The target grid geometry
    filepath_geomvars = '/bettik/ockendeh/NEMO_simulations/NEMO025_bmach_geom_vars.nc'
    # The eORCA1 bedmachine mask
    filepath_eORCA1_bmach_masks = '/bettik/ockendeh/NEMO_simulations/eORCA1_bmach_masks.nc'
    # Neural network filepaths 
    path_model = '/bettik/ockendeh/NEMO_simulations/models/'
    path_norm_metrics = '/bettik/ockendeh/NEMO_simulations/models/'
    filepath_domain_cfg = '/bettik/ockendeh/NEMO_simulations/TGCC/eORCA1.4.3_CavsForNN_domain_cfg.nc'

# And the location of the output files 
# filepath_nn_output = '/ccc/scratch/cont003/gen6035/ockendeh/NEMO/eORCA1.L75/eORCA1.L75-TOTO-RST.007/'

# Any parameters which need to be set 
# Which neural network have you chosen?
this_collection = 'OPM026_OPM0263_OPM031_OPM016_OPM018_OPM021_ctrl94_isf94_isfru94'
exp_name = 'slope_front'
join_ice_shelves = True


# Apply the neural network 
apply_nn_to_NEMO(filepath_nemo_output, 
                     filepath_mask, 
                     filepath_geomvars, 
                     filepath_eORCA1_bmach_masks, 
                     path_model, 
                     path_norm_metrics, 
                     filepath_nn_output, 
                     this_collection, 
                     exp_name, 
                     join_ice_shelves)
