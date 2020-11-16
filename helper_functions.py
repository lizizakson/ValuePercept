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

#print("oren")

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