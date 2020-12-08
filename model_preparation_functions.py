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

def equalize(X_features, df, subject_IDs):
    """
    Equalize number of subjects in x_features and y (behavior).

    input: x_features (a numpy array which contains the x_features of all subjects), behavior (df with behavior results), 
    subject_IDs (list of subject IDs from the fMRI data (x_features) in an integer form)

    output: x_features and behavior df with the same amount of subjects
    """
    #Exclude subjects in the behavior data that are not in the fMRI data
    df = df[df['Subject'].isin(subject_IDs)]

    print("Before equalization:")
    print("Number of subjects in x_features is {0}".format(len(subject_IDs)))

    subjectBehavior = df['Subject']
    print("And, the number of subjects in the behavior data is {0}".format(len(subjectBehavior)))

    #difference between behavior and fMRI data sets
    subjectDifference = np.setdiff1d(subject_IDs, subjectBehavior)
    print("So, the difference between the two data sets is {0}".format(len(subjectDifference)))

    #Exclude the subjects that are not in the behavior data from the fMRI data
    id_to_array = dict(zip(subject_IDs,range(len(subject_IDs))))

    X_features = np.delete(X_features,[id_to_array[sub] for sub in subjectDifference], axis=0)

    #Delete the excluded subjects from the subjects_IDs
    subject_IDs = np.delete(subject_IDs,[id_to_array[sub] for sub in subjectDifference], axis=0)
    

    print("After equalization:")
    print("Number of subjects in x_features is {0}".format(len(subject_IDs)))

    
    print("And, the number of subjects in the behavior data is {0}".format(len(subjectBehavior)))

    #difference between behavior and fMRI data sets
    subjectDifference = np.setdiff1d(subject_IDs, subjectBehavior)
    print("So, the difference between the two data sets is {0}".format(len(subjectDifference)))

    return X_features, df, subject_IDs


def vectorize_mat(X_features, subject_IDs):
    """
    Produce vectorized_correlation = reduce the matrix to half and discrad diagonal - for all subjects

    input: x_features (a numpy array which contains the x_features of all subjects), subject_IDs (a list of the subjects in the fMRI data)
    output: vectorized mats in the form of pandas df, each row contains the identity of this specific edge
    """
    num_nodes = X_features.shape[1] # num_nodes (integer with the number of nodes of the matrices)

    vectorized_edges = {}
    for node1 in range(num_nodes):
        for node2 in range(node1+1,num_nodes):
            vectorized_edges[(node1,node2)] = []
            for i in range(len(X_features)):
                vectorized_edges[(node1,node2)].append(X_features[i][node1,node2]) #append the edge of this specific pair of nodes

    print("The number of vectorized edges is {0}".format(len(vectorized_edges)))
    print("Make sure it equals num_nodes*(num_nodes-1)/2")

    #first display: rows are the edges identity and columns are subject IDs
    vectorized_edges = pd.DataFrame.from_dict(vectorized_edges, orient='index', columns = subject_IDs)

    #second display: transpose the data frame so that the subjects would be the rows and the edges identity the columns
    vectorized_edges = vectorized_edges.transpose()
    print(vectorized_edges.head(5)) #print this display

    return vectorized_edges

