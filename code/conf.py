# This is a config file. Please fill out the correct paths.

#These paths are for the PreProcessing step (only if needed):
ZIPPED_RAW_DATA_DIR = "/mnt/d/HCP_Liz/zipped_data" #This is a large dataset which requires a lot of space. I kept it in an external HD
UNZIPPED_RAW_DATA_DIR = "/mnt/d/HCP_Liz/zipped_data" 
CONCATENATED_DATA_DIR = "/mnt/d/HCP_Liz/concatenated_data/"

#These paths are for the Model step (required to have the preprocessing outputs):
BASE_DIR = '/mnt/c/Users/liz/Contacts/Desktop/ValuePercept/'

# Pre-processed RS data = x_features
X_FEATURES_DIR = 'RS_data/FC_data_processed/'
X_FEATURES_FILENAME = 'all_features_992_Shafer100_7N.npz'
SUB_ID_FILENAME = 'subID_992.txt'

# Behavior = y
BEHAVIOR_PATH = 'Behavior/HCP_behavior_includeZscores_V2.csv'