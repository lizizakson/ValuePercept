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
from dypac import Dypac
import nilearn
from nilearn import image, plotting, datasets
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.decomposition import DictLearning, CanICA
from scipy.stats import pearsonr
import nilearn.plotting as plotting
import csv
import scipy as sp
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from numpy import arange
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


import feature_selec_functions
import model_preparation_functions
from model_functions import Model
import conf
import importlib


def main():
    #load data for model
    df = pd.read_csv('df_behav.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print(df.shape)
    all_fc_data = pd.read_csv('all_fc_data.csv')
    all_fc_data = all_fc_data.loc[:, ~all_fc_data.columns.str.contains('^Unnamed')]
    print(all_fc_data.shape)
    
    #pre-provessing
    params_dict = {'X_preprocess_type': 'fisher_z', 'y_preprocess_type': None, 'test_size': 0.2}
    behav_name = "DDisc_AUC_40K"
    behav = df[behav_name] #send the behavioral score: DDisc_AUC_40K, Flanker_AgeAdj, rank_compare

    reg_flanker = Model('ElasticNet', params_dict, all_fc_data, behav)
    reg_flanker.preprocess()
    reg_flanker.split_train_test(dbg = False)

    #fit
    params_run = {'params_selec': {"selection_method": "univariate", "corr_type": "pearson", "percent": 0.05}, 
                    'params_fit': {'l1_ratio': 0.01, 'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0]},
                    'params_predict': {'eval_score': 'mean_squared_error', 'best_params': None}}
    
    final_model, final_model_score, yhat, rel_features, rel_features_temp = reg_flanker.fit_model(override_params = params_run)
    print(final_model)

    #output
    colors_dict = {'DDisc_AUC_40K': '#5eaaa8', 'Flanker_AgeAdj': "#eb5e0b", 'rank_compare': "#4a47a3", 'control_task': "#FF3659"}
    parc_file_name = "Schaefer2018_100Parcels_7Networks_order_info.txt"
    
    reg_flanker.plot_output(yhat, colors_dict[behav_name]) #color according to the specific predicted task scores

    save_name = behav_name + ".csv"
    reg_flanker.save_rel_features(final_model, rel_features, parc_file_name, save_name)
    reg_flanker.set_rel_features(rel_features_temp)
    #print(reg_flanker.rel_features)

    #cross_model
    behav_name = "Flanker_AgeAdj"
    behav = df[behav_name]
    reg_DD = Model('ElasticNet', params_dict, all_fc_data, behav)
    reg_DD.set_rel_features(reg_flanker.rel_features)
    reg_DD.preprocess()
    reg_DD.split_train_test(dbg = False)
    #fit
    params_run = {'params_fit': {'l1_ratio': 0.01, 'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0]},
                    'params_predict': {'eval_score': 'mean_squared_error', 'best_params': None}}
    
    final_model, final_model_score, yhat = reg_DD.fit_model(override_params = params_run, select_features = False)
    print(final_model)
    reg_DD.plot_output(yhat, colors_dict[behav_name])

if __name__ == "__main__":
    main()

#'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]