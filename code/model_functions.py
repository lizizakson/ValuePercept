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
import scipy as sp
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from numpy import arange
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet

import hcp_utils as hcp
import feature_selec_functions
import model_preparation_functions
import conf
import importlib
import plots

class Model:
    def __init__(self, model_name, params_dict):
        self.model_name = model_name
        self.params_dict = params_dict
        self.X_df = None
        self.y_df = None

    #Pre-process functions:
    def preprocess(self, all_fc_data, behav):
        """
        Apply pre-processing to the x_data and y_data.
        input: all_fc_data = a df of the x_features, behav = a pandas series of behavioral scores
        output: a df of modified x_features according wanted params (fisher_z/scaled/both/none) and a df of modified y 
        """
        _prepare_X(all_fc_data)
        _prepare_y(behav)

    def _prepare_y(self, behav):
        """
        Modifies the y according to the wanted params.
        Either none or saling.
        input: behav = a pandas series of behavioral scores
        output: y_df = a pandas series of modified behavioral scores
        """
        self.y_df = behav.copy()
        self.params_dict["y_preprocess_type"] == "z_score":
            self.y_df = self.y_df.transform(lambda x: stats.zscore(x))

    def _prepare_X(self, all_fc_data):
        """
        Modifies the x_features according to the wanted params.
        Either fisher-z transformation, or z-scoring, or both.
        input: all_fc_data = a df of the x_features, possible modifications ("fisher_z"/"z_score"/"both").
        output: X_df = a df of modified x_features according to the modify variable.
        """
        self.X_df = all_fc_data.copy()
        if self.params_dict["X_preprocess_type"] == "fisher_z": #Fisher z_transform of x_features
            self.X_df = self.X_df.transform(lambda x: np.arctanh(x))
        elif self.params_dict["X_preprocess_type"] == "z_score": #scaling of x_features
            self.X_df = self.X_df.transform(lambda x: stats.zscore(x))
        elif self.params_dict["X_preprocess_type"] == "both": #first fisher z_transform, then z_score
            self.X_df = self.X_df.transform(lambda x: np.arctanh(x))
            self.X_df = self.X_df.transform(lambda x: stats.zscore(x))

        return

    def get_x(self):
        return self.X_df.copy()

    def set_x(self, other_df):
        self.X_df = other_df.copy()

    def split_train_test(self):
        """
        Splits the data into train and test datasets according to the test_size variable.
        Also shuffles the data.
        """
        X_train, X_test, y_train, y_test = train_test_split(
                                                            self.X_df, # x
                                                            self.y_df, # y
                                                            test_size = params_dict["test_size"], # e.g. 60%/40% split  
                                                            shuffle = True, # shuffle dataset
                                                                            # before splitting
                                                            random_state = 0 # same shuffle each
                                                                            # time
                                                                            )

        # print the size of our training and test groups
        print('training:', len(X_train),
            'testing:', len(X_test))
        
        #Examine the distributions of y_train and y_test
        sns.distplot(y_train,label='train')
        sns.distplot(y_test,label='test')
        plt.legend()

        return X_train, X_test, y_train, y_test

    def do_cv(self, nfolds = 5):
        # bla bla
        assert(0)
        return

    #Fit functions:
    def override_params(self):
        if override_params == None:
            override_params = self.params_dict.copy()

        for k in self.params_dict.keys():
            if k in override_params.keys():
                continue
            override_params[k] = self.params_dict[k]

    def fit(self, override_params = None):
        # apply override_params
        _select_features_uni(X_train) #select features
        _select_features_multi(X_train)
        _arrange_selected_data(data, mask_dict, corr_type = "pearson") #filter x_train according to the selected features
        _optimized_cv(X_train, y_train)

    def _select_features_uni(X_train, y_train):
        """
        Runs the CPM feature selection step (univariate feature selection) which is written under feature_selec_functions: 
        - correlates each edge with behavior, and returns a mask of the X% highest correlated edges (with behavior)

        input: x features of train/test, y train/test (behavior), % of highest correlated edges, corr type
        output: a mask of the X% highest correlated edges (with behavior)
        """ 
        importlib.reload(feature_selec_functions) #reload any changes in the code
        mask_dict, corr = feature_selec_functions.select_features(X_train, y_train, percent = 0.05, corr_type='pearson', verbose=False)    
        
    def _arrange_selected_data(data, mask_dict, corr_type = "pearson"):
        if corr_type == 'pearson':
            X_train_selec = data[mask_dict.index[mask_dict]]
            print("The X_train_selec (pearson) shape is {}".format(X_train_selec.shape))
        elif corr_type == 'spearman':
            #spearman corr - because the are no tupples as indices
            mask_dict_list = mask_dict.index[mask_dict].to_list()
            X_train_selec = data[data.columns[mask_dict_list]]
            print("The X_train_selec (spearman) shape is {}".format(X_train_selec.shape))
        return X_train_selec

    def optimized_cv(X_train, y_train, model_type):
        """
        Runs a grid search to tune the hyper-parameters of the model.
        input: X_train, y_train, model = either 'ElasticNet'/ 'SVR'/ 'RandomForest'
        output: best score and best parameters
        """
        # define model evaluation method
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # define grid
        grid = dict()
        if model_type == 'ElasticNet':
            print("first if")
            model = ElasticNet(l1_ratio = 0.01)
            grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
        elif model_type == 'SVR':
            model = SVR(kernel = 'linear')
            grid['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
        elif model_type == 'RandomForest':
            model = RandomForestRegressor()
            grid['max_depth'] = range(3,7)
            grid['n_estimators'] = (10, 50, 100, 1000)
            
        # define search
        search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

        # perform the search
        results = search.fit(X_train, y_train)

        # summarize
        print('MAE: %.3f' % results.best_score_)
        print('Config: %s' % results.best_params_)
        
        return results, results.best_params_