# This is a config file. Please fill out the correct paths.

#These paths are for the PreProcessingZippedFiles step (only if needed):
ZIPPED_RAW_DATA_DIR = "/mnt/d/HCP_Liz/zipped_data" #This is a large dataset which requires a lot of space. I kept it in an external HD
UNZIPPED_RAW_DATA_DIR = "/mnt/d/HCP_Liz/zipped_data" 
CONCATENATED_DATA_DIR = "/mnt/d/HCP_Liz/concatenated_data/"

#These paths are for the PreProcessingRSdata step (only if needed):
SCHAEFER_PARC_DIR = "/mnt/c/Users/liz/Contacts/Desktop/ValuePercept/Parcellations/Parcellations/HCP/fslr32k/cifti/"
SCHAEFER_PARC_FILE = "Schaefer2018_100Parcels_17Networks_order.dlabel.nii"
MEDIAL_MASK_PATH = "/mnt/c/Users/liz/Contacts/Desktop/ValuePercept/Parcellations/Parcellations/fs_LR_32k_medial_mask.mat"

#These paths are for the Model step (required to have the preprocessing outputs):
BASE_DIR = '/mnt/c/Users/liz/Contacts/Desktop/ValuePercept/'

## Pre-processed RS data = x_features
X_FEATURES_DIR = 'RS_data/FC_data_processed/'
X_FEATURES_FILENAME = 'all_features_992_Shafer100_7N.npz'
SUB_ID_FILENAME = 'subID_992.txt'

## Behavior = y
BEHAVIOR_PATH = 'Behavior/HCP_behavior_includeZscores_V2.csv'

#These paths are for the Plot Results step (required to have the Model outputs):
RESULTS_DIR = '/mnt/c/Users/liz/Contacts/Desktop/ValuePercept/results/'

#Paths for ROI analyses:
## Value network paths:
VALUE_MASK_PATH = BASE_DIR + 'ROI_analysis/1-s2.0-S1053811913002188-mmc2/nifti_masks/fig09/'
vmPFC_mask_file = VALUE_MASK_PATH + 'binConjunc_PvNxDECxRECxMONxPRI_striatum.nii.gz'
STR_mask_file = VALUE_MASK_PATH + 'binConjunc_PvNxDECxRECxMONxPRI_vmpfc.nii.gz'