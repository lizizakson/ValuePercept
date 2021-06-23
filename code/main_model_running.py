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


def main(cross = False, split_fam = True):
    #load data for model
    df = pd.read_csv('df_behav_fam.csv', index_col=0) #subject ID as index column
    print(df.shape)
    all_fc_data = pd.read_csv('all_fc_data.csv', index_col=0) #subject ID as index column
    print(all_fc_data.shape)
    
    params_dict = {'X_preprocess_type': 'z_score', 'y_preprocess_type': None, 'test_size': 0.2}
    #df['rank_compare_log'] = np.log(df['rank_compare'])
    #df['rank_compare_boxcox'], fitted_lambda = stats.yeojohnson(df['rank_compare'])
    behav_name = "Flanker_AgeAdj"

        

    #make sure that the neural data (all_fc_data) and the behavioral data are the same length
    if df[behav_name].isnull().sum() > 0: #if there are missing values in the behavioral measurement
        df = df.dropna(subset=[behav_name]) #exclude the null score from the df

        #equalize the subjects in the behav data and the fc data
        subject_IDs = all_fc_data.index
        subjectBehavior = df.index
        print("Before equalization:")
        print("Number of subjects in x_features is {0}".format(len(subject_IDs)))
        print("And, the number of subjects in the behavior data is {0}".format(len(subjectBehavior)))

        subjectDifference = np.setdiff1d(subject_IDs, subjectBehavior)
        print("So, the difference between the two data sets is {0}".format(len(subjectDifference)))

        #make sure the difference is smaller than 5
        assert len(subjectDifference) < 5, "large difference between behav and FC data sets"

        all_fc_data = all_fc_data.drop(subjectDifference)
    
    #Make sure the subject lists in the fMRI data and behavior data are equal
    assert all_fc_data.index.equals(df.index), "Row (subject) indices of FC vcts and behavior don't match!"
        
    behav = df[behav_name] #send the behavioral score: DDisc_AUC_40K, Flanker_AgeAdj, rank_compare

    #pre-processing
    reg_flanker = Model('ElasticNet', params_dict, all_fc_data, behav, df)
    reg_flanker.preprocess()

    if split_fam:
        reg_flanker.split_train_test_fam(dbg = True)
    else:
        reg_flanker.split_train_test(dbg = False)

    #fit
    params_run = {'params_selec': {"selection_method": "univariate", "corr_type": "pearson", "percent": 0.05}, 
                    'params_fit': {'l1_ratio':  [0.01], 'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0]},
                    'params_predict': {'eval_score': 'mean_squared_error', 'best_params': None}}
    
    final_model, final_model_score, yhat, rel_features, rel_features_temp = reg_flanker.fit_model(override_params = params_run)
    print(final_model)

    #output
    colors_dict = {'DDisc_AUC_40K': '#5eaaa8', 'Flanker_AgeAdj': "#eb5e0b", 'rank_compare_boxcox': "#4a47a3", 
                    'PicVocab_AgeAdj': "#FF3659", "Mars_Final": "#808080"}
    parc_file_name = "Schaefer2018_100Parcels_7Networks_order_info.txt"
    
    reg_flanker.plot_output(yhat, colors_dict[behav_name]) #color according to the specific predicted task scores

    save_name = behav_name + ".csv"
    reg_flanker.save_rel_features(final_model, rel_features, parc_file_name, save_name)
    reg_flanker.set_rel_features(rel_features_temp)
    #print(reg_flanker.rel_features)

    #cross_model
    if cross:
        behav_name = "DDisc_AUC_40K"
        behav = df[behav_name]
        reg_DD = Model('ElasticNet', params_dict, all_fc_data, behav)
        reg_DD.set_rel_features(reg_flanker.rel_features)
        reg_DD.preprocess()
        reg_DD.split_train_test(dbg = False)
        #fit
        params_run = {'params_fit': {'l1_ratio': np.arange(0, 1, 0.01), 'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0]},
                        'params_predict': {'eval_score': 'mean_squared_error', 'best_params': None}}
        
        final_model, final_model_score, yhat = reg_DD.fit_model(override_params = params_run, select_features = False)
        print(final_model)
        reg_DD.plot_output(yhat, colors_dict[behav_name])

if __name__ == "__main__":
    main()

#'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]