import os
import re

import shutil
import codecs, json 
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
import hcp_utils as hcp

import conf
import helper_functions
import importlib

def binarize_mask(mask, thr):
    """
    Binarize mask according to a threshold.
    input: an np array which comprise of a lot of numbers, and a threshold (=int).
    output: an np array in the same size which comprise of 0 and 1.
    """
    mask[mask < thr] = 0 #smaller than the thr is 0.
    mask[mask > thr] = 1 #larger than the thr is 1.
    return mask

def masks_union(masks):
    """
    Combine all components of ICA to one mask.
    input: an np array in which each row is a component and the columns are the vertices.
    output: an np array with one row which is a combination of all the components.
    """
    #mask_sum = np.empty([masks.shape[1]],) #number of rows = number of vertices
    #print(masks.shape[0])
    for i in range(masks.shape[0]): #loop over number of components
        mask_sum += masks[i]
    
    return mask_sum
 
def idx_mask(mask):
    """
    Show indices of vertices according to a mask.
    input: a binary mask.
    output: a list of relevant indices according to the binary mask (the indices where the mask equals 1).
    """
    #mask = mask.flatten().tolist()
    #mask_int = [int(i) for i in mask]
    #mask_bin = np.array(mask_int)

    result = np.where(mask == 1)
    idx = list(result[0])
    return idx 

