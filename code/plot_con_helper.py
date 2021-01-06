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
import feature_selec_functions
import csv

import hcp_utils as hcp

import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle


def sync_specific_df(df_dict, df_key):
    """
    Helps to the equalize_parc function:
    For every df, checks if all pair of nodes that exist in both tasks exist in the df. 
    if not, append it to the df with a beta of np.nan.
    input: a dict of dfs, and a specific key (=specific df)
    output: the specific df with equalized parcels (according to the other dfs in the dict)
    """
    df_res = df_dict[df_key]
    for key in df_dict.keys(): #loop over all other dfs
        if key == df_key:
            continue

        for i in range(df_dict[key].shape[0]): #loop over all pairs of parcels (nodes) in the other df
            node1_val = df_dict[key].loc[i]['node1'] #specific value of node 1 according to a row number
            node2_val = df_dict[key].loc[i]['node2']
            #Searches if this pair exists in the specific df we are looking at
            res_tmp = df_dict[df_key][(df_dict[df_key]['node1'] == node1_val) & (df_dict[df_key]['node2'] == node2_val)]
            if res_tmp.shape[0] == 0: #this pair does not exist
                df_res = df_res.append({'node1': node1_val, 'node2': node2_val, 'beta': np.nan}, ignore_index=True)

    return df_res


def equalize_parc(df_dict):
    """
    For every df, checks if all pair of nodes that exist in both tasks exist in the df. 
    if not, append it to the df with a beta of np.nan.
    input: a dict of all the dfs (each df contains: node1, node2, and beta)
    output: a dict of all the dfs with equalized parcels
    """
    res_df_dict = {}

    for key in df_dict.keys():
        #print(key)
        df = sync_specific_df(df_dict, key)
        res_df_dict[key] = df
    
    return res_df_dict


def unique(list1): 
    x = np.array(list1) 
    return list(np.unique(x)) 

def plot_con_helper(df):

    """
    Helps in the plot_connectome step:
    input: a df of node1, node2 and beta
    output: a list of node names for this specific df=node_names, and an np array of abs(beta)=con
    """
    
    node_names = list(df.node1) + list(df.node2)
    node_names = list(set(node_names))
    con = df.beta

    i_1 = []
    i_2 = []
    for f in range(len(df)):
        feature = df.iloc[f,:]
        i_1.append(node_names.index(feature['node1']))
        i_2.append(node_names.index(feature['node2']))
    indices = (np.array(i_1), np.array(i_2))

    lh_names = [name for name in node_names if 'LH' in name]
    rh_names = [name for name in node_names if 'RH' in name]
    lh_ordered = []
    rh_ordered = []
    net_names = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
    for network in net_names: 
        lh_net_names = [name for name in lh_names if network in name]
        lh_ordered = lh_ordered + lh_net_names

        rh_net_names = [name for name in rh_names if network in name]
        rh_ordered = rh_ordered + rh_net_names
    lh_ordered.reverse()
    ordered = lh_ordered + rh_ordered

    layout = circular_layout(node_names=node_names, node_order=ordered, group_boundaries=[0, len(lh_ordered)])
    colors_list = ['purple', 'blue', 'green', 'violet', 'wheat', 'orange', 'red']
    #colors_list = [(120/255, 18/255, 133/255), (70/255, 130/255, 180/255), (0/255, 118/255, 14/255), (196/255, 57/255, 248/255), (220/255, 248/255, 162/255), (230/255, 146/255, 32/255), (204/255, 60/255, 78/255)]

    color_dict = dict(zip(net_names, colors_list))
    node_colors = []
    for name in node_names:
        net = name.split('_')[1]
        node_colors.append(color_dict[net])

    return node_names, con, indices, ordered, node_colors, layout

def assign_nets(df, col_name):
    list_res = []
    for val in df[col_name]:
        net = val.split('_')[1]
        list_res.append(net)

    return list_res


def add_col(df):
    """
    Add columns to df: beta_abs, node1_networks, and node2_networks
    """
    #Add column of abs_beta
    df['beta_abs'] = np.abs(df.beta)
    df_arranged = df

    #Create two extra columns of node1_nets and node2_nets
    df_arranged['node1_nets'] = assign_nets(df_arranged, 'node1')
    df_arranged['node2_nets'] = assign_nets(df_arranged, 'node2')

    return df_arranged



def compare_edges(df_arranged, df_nodes):
    """
    Find the common edges between 2 data frames of edges which predict tasks.
    input: 2 data frames which contain node1, node2 and beta
    output: a list of common edges
    """
    
