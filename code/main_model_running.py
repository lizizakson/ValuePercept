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
    behav = df["DDisc_AUC_40K"] #send the behavioral score

    reg_flanker = Model('ElasticNet', params_dict, all_fc_data, behav)
    reg_flanker.preprocess()
    reg_flanker.split_train_test(dbg = False)

    #fit
    params_run = {'params_selec': {"selection_method": "univariate", "corr_type": "pearson", "percent": 0.05}, 
                    'params_fit': {'l1_ratio': 0.01, 'alpha': [1e-2, 1e-1, 0.0, 1.0]},
                    'params_predict': {'eval_score': 'mean_squared_error', 'best_params': None}}
    
    reg_flanker.fit_model(override_params = params_run)

if __name__ == "__main__":
    main()

#'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]