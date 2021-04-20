import os
import re

import shutil
import codecs, json 
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
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
import csv

import hcp_utils as hcp
import conf

import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle
import plot_con_helper
import feature_selec_functions
import importlib

def conv_to_tuple(temp_list):
    list_of_tups = []
    for i in temp_list:
        list_of_tups.append(eval(i))
    return list_of_tups

def arrange_df(all_features, model_features, model_beta):
    #create model_features_logic and beta for df
    model_features_logic = []
    beta = []
    i = 0
    for f in all_features:
        if f in model_features:
            model_features_logic.append(1)
            beta.append(model_beta[i])
            i += 1
        else:
            model_features_logic.append(0)
            beta.append(np.nan)
     
    #create df
    df = pd.DataFrame(columns=['all_features', 'model_features_logic', 'beta'])
    df["all_features"] = all_features
    df["model_features_logic"] = model_features_logic
    df["beta"] = beta

    return df

def get_node_names(features, list_parcel):
    res = [] #node_names
    for i in range(len(features)):
        dict_res = {}
        tup = features[i] #tupple(pair of nodes)
        parcel_index = tup[0] #tupples and list_parcels are listed from zero
        dict_res["node1"] = list_parcel[parcel_index]
        parcel_index = tup[1]
        dict_res["node2"] = list_parcel[parcel_index]
        res.append(dict_res)
    #turn res into a df:
    df_res = pd.DataFrame(res)
    return df_res

def add_node_names(df, node_names):
    left_index = df.index
    df_with_node_names = pd.merge(df, node_names, on = left_index, left_index = True, right_index = True)
    return df_with_node_names
    
    


