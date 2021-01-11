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


def ind_Pmaxelements(list1, n_features, P):

    """
    Helps in the feature selection step: 
    input: a list of numbers (edges)
    output: the threshold (the minimum maximum) according to a specified percentage of highest values
    """
    num = P*n_features
    list_temp = list(map(abs, list1)) #convert all elements to absolute value

    ind = np.argpartition(list_temp, -num)[-num:] #indices of P highest values
    print(ind)
     
    return ind


def Pmaxelements(list1, P):

    """
    Helps in the feature selection step: 
    input: a list of numbers (edges)
    output: the threshold (the minimum maximum) according to a specified percentage of highest values
    """
    
    #list1 = list1.flatten() #make sure the array is 1d

    list_temp = list(map(abs, list1)) #convert all elements to absolute value
   
    list_temp.sort(reverse = True) #sort the elements from the highest to the lowest element
    
    num_elements = int(len(list_temp)*P) #num of highest elements needed to be found according the % given
    
    final_list = list_temp[0:num_elements] #take the highest X elements from the original list
    
    threshold = final_list[num_elements-1] #the minimum highest element in the list (-1 because the index starts from 0)
    
    return threshold


def select_features(train_vcts, train_behav, percent = 0.01, corr_type='pearson', verbose=False):
    
    """
    Runs the CPM feature selection step: 
    - correlates each edge with behavior, and returns a mask of the X% highest correlated edges (with behavior)

    input: x features of train/test, y train/test (behavior), % of highest correlated edges, corr type
    output: a mask of the X% highest correlated edges (with behavior)
    """

    assert train_vcts.index.equals(train_behav.index), "Row indices of FC vcts and behavior don't match!"

    # Correlate all edges with behav vector
    if corr_type =='pearson':
        cov = np.dot(train_behav.T - train_behav.mean(), train_vcts - train_vcts.mean(axis=0)) / (train_behav.shape[0]-1)
        corr = cov / np.sqrt(np.var(train_behav, ddof=1) * np.var(train_vcts, axis=0, ddof=1))
    elif corr_type =='spearman':
        corr = []
        for edge in train_vcts.columns:
            r_val = sp.stats.spearmanr(train_vcts.loc[:,edge], train_behav)[0]
            corr.append(r_val)
    
    # Correlate all edges with behav vector
    #corr = []
    #for (columnName, columnData) in train_vcts.iteritems():
    #print(sc.stats.pearsonr(x, columnData.values)[0])
     #   corr.append(sc.stats.pearsonr(train_behav, columnData.values)[0])  
    #print(corr)
    ##This code gives the same result as the code for the corr analysis above!

    # threshold of the X % highest values
    threshold = Pmaxelements(corr, percent)
    print(threshold)

    # define mask according to threshold
    mask_edges = [] #create new list to store the indices of the highest elements from the original list in
    #corr.abs() #convert the original list to abs values (not sorted)
    mask_edges = corr.abs() >= threshold
    print(corr[mask_edges == True])

    if verbose:
        print("Found ({}) edges positively/negatively correlated with behavior in the training set".format(len(mask_edges))) # for debugging

    print(len(corr))
    #print(len(mask_dict))
    return mask_edges, corr




