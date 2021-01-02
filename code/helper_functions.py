import os, zipfile
import re

import shutil
import codecs, json 
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import scipy as sp
from scipy import stats
import sklearn as sk
import time
from dypac import Dypac
import nilearn
from nilearn import image, plotting, datasets
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.decomposition import DictLearning, CanICA
from scipy.stats import pearsonr
import nilearn.plotting as plotting
import math

from nilearn import connectome
import hcp_utils as hcp

def unzip(dir_path):
    """
    This function loops over the files (a file for each subject) in the directory, unzips the zip files, closes the files,
    and deletes the original zip files

    input: directory path

    """
    extension = ".zip"

    os.chdir(dir_path) # change directory from working dir to dir with files

    for item in os.listdir(dir_path): # loop through items in dir
        if item.endswith(extension): # check for ".zip" extension
            file_name = os.path.abspath(item) # get full path of files
            zip_ref = zipfile.ZipFile(file_name) # create zipfile object
            zip_ref.extractall(dir_path) # extract file to dir
            zip_ref.close() # close file
            os.remove(file_name) # delete zipped file



def concat(dir_path, sessions, file_name,saved_dir_path):
    """
    This function loops over the files in the directory (a file for each subject): each subject has 4 runs.
    The function loads the specific file of timeseries that is specified in the input for each run, normalize the data 
    in each run, and concatenate the 4 runs per subject.
    Also, the function deletes the original subject file (to save space).

    input: directory path, sessions (a list with the runs/sessions names), file name (the specific file of timeseries which we 
    are interested to analyze, write it without the session name), saved_dir_path (where to save the concatenated files)

    output: a file which contains the 4 runs normalized and concatenated in the diretory
    """
    subjects = os.listdir(dir_path) #creates a list of subject numbers
    #Loop over each subject and each session per subject
    for sub in subjects:
        #if sub in data.keys():
        #   continue
        tmp_dict = {}
        for session in sessions:
            path = '{0}/{1}/MNINonLinear/Results/rfMRI_REST{2}/rfMRI_REST{2}{3}'.format(dir_path,sub,session,file_name)
            #load data per session
            img = nib.load(path)
            X = img.get_fdata()
            if X.shape[0] != 1200: #validate that each run is 1200 time points
                print("bad subject")
            #Xn = X - np.mean(X) #remove the mean image (?)
            Xn = hcp.normalize(X) #normalize data per session
            tmp_dict[session] = Xn
        #Concatenate 4 normalized sessions per subject
        data_sub = np.concatenate((tmp_dict[sessions[0]], tmp_dict[sessions[1]],tmp_dict[sessions[2]], tmp_dict[sessions[3]]), axis=0)
        print("finished subject {0}".format(sub), flush = True)
        #data_sub
        #save file of subject to the directory
        np.savez_compressed(saved_dir_path + sub, a = data_sub)
        #delete original subject file
        #shutil.rmtree(dir_path + '/' + sub, ignore_errors=False, onerror=None) 
        #break
    #data


def medial_mask(data, medial_mask):
    """
    This function gets rsfMRI data file (numpy array) and a mask file (mat file that was converted to array), and produces data file with the medial mask inserted into it (numpy array).

    input: data of rsfMRI per subject, medial mask (1: 59412 cortical vertices- same as HCP data, 0: 5572 medial vertices that have to be inserted into the HCP data)
    output: HCP data with 59412 cortical vertices and 5572 new medial vertices as NaNs inserted into it
    """
    #Take the first 59412 vertices from the HCP data. These are the cortical vertices (the Shafer parcellation doesn't include subcortical areas)
    data_forparc = data[:,0:59412] #0-59411 (the last element is excluded)

    #indices in which the nans should be inserted into
    result = np.where(medial_mask == 0)
    idx = list(result[0])

    #fix the indices such that the insertion would be correct (substruct the index's position from each index)
    for i in range(len(idx)):
        idx[i] = idx[i] - i

    #insert NaNs into the indices of the HCP data according to the medial_mask
    idx = tuple(idx)
    data_forparc = np.insert(data_forparc, idx, np.nan, axis =1)
    print(data_forparc.shape)

    #Validate the output
    idx_correct = list(result[0])
    for i in idx_correct:
        if not math.isnan(data_forparc[0,i]):
            print("problem")

    return data_forparc



def schaefer_parc(data_forparc, parc):
    """
    This function gets rsfMRI data file (numpy array) and a parcellation file (dscalar/dlabel), and produces parcellated data (numpy array).

    input: data of rsfMRI per subject, parcellation file
    output: parcellated data (in the shape: num of time points, num of parcels)
    """
    #path = "/mnt/c/Users/liz/Contacts/Desktop/ValuePercept/Parcellations/Parcellations/HCP/fslr32k/cifti/Schaefer2018_100Parcels_7Networks_order.dlabel.nii"
    #img = nib.load(path)
    #parc = img.get_fdata()
    parc = parc.astype(int) #Convert parcel numbers from float to integer

    #insert the parellation numbers as column names to the data
    data_forparc = pd.DataFrame(data=data_forparc, columns=parc[0])
    
    #Initialize the parc_ts (=parcellated data)
    shape = (data_forparc.shape[0], max(np.unique(parc))) #4800,100
    parc_ts = np.zeros(shape)
    
    #Average across the same parcell (e.g. all the columns(vertices) who belong to parcel 1) for each row (time point)
    #Start from 1, since 0 is has no maening in this parcellation
    for i in range(1,max(np.unique(parc))+1): #1-101, last elemnt is excluded in python
        parc_ts[:, i-1] = data_forparc[i].mean(axis=1) #i-1 so it will fit into 0-99 columns in the numpy array output

    print(parc_ts.shape)

    return parc_ts


def create_all_features(data, parc):
    """
    This function gets the concatented data and produces a connectiviry matrix according to the chosen parcellation for each subject.
    input: data = a list of paths for the concatenated data (a poth for each subject), parc = chosen parcellation method
    output: all_features = a list which contains all the connectivity matrices
    """
    all_features = [] # here is where we will put the data (a container)

    #Create the wanted connectivity matrix 
    #kind{“correlation”, “partial correlation”, “tangent”, “covariance”, “precision”}
    correlation_measure = connectome.ConnectivityMeasure(kind='correlation')

    for i,sub in enumerate(data):
        #load data
        load_data = np.load(data[i])['a']
        # parcell the data according to a specific atlas
        parcellated_data = (hcp.parcellate(load_data, parc))
        # create a region x region correlation matrix
        correlation_matrix = correlation_measure.fit_transform([parcellated_data])[0]
        # add to our container
        #np.savez_compressed(saved_dir_path + i, a = correlation_matrix)
        all_features.append(correlation_matrix)
        # keep track of status
        print('finished %s of %s'%(i+1,len(data)))

    return all_features
