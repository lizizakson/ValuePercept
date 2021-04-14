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
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import copy

import hcp_utils as hcp
import feature_selec_functions
import model_preparation_functions
import conf
import importlib
import plots

class Model:
    def __init__(self, model_name, params_dict, X_df, y_df):
        self.model_name = model_name
        if params_dict is None:
            self.fill_default_params() #need to write
        else:
            self.params_dict = params_dict
        self.X_df = X_df
        self.y_df = y_df
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        

    #Pre-process functions:
    def _prepare_y(self):
        """
        Modifies the y according to the wanted params.
        Either none or saling.
        input: behav = a pandas series of behavioral scores
        output: y_df = a pandas series of modified behavioral scores
        """
        self.y_df = self.y_df.copy()
        if self.params_dict["y_preprocess_type"] == "z_score":
            self.y_df = self.y_df.transform(lambda x: stats.zscore(x))

        return self.y_df

    def _prepare_X(self):
        """
        Modifies the x_features according to the wanted params.
        Either fisher-z transformation, or z-scoring, or both.
        input: all_fc_data = a df of the x_features, possible modifications ("fisher_z"/"z_score"/"both").
        output: X_df = a df of modified x_features according to the modify variable.
        """
        self.X_df = self.X_df.copy()
        if self.params_dict["X_preprocess_type"] == "fisher_z": #Fisher z_transform of x_features
            self.X_df = self.X_df.transform(lambda x: np.arctanh(x))
        elif self.params_dict["X_preprocess_type"] == "z_score": #scaling of x_features
            self.X_df = self.X_df.transform(lambda x: stats.zscore(x))
        elif self.params_dict["X_preprocess_type"] == "both": #first fisher z_transform, then z_score
            self.X_df = self.X_df.transform(lambda x: np.arctanh(x))
            self.X_df = self.X_df.transform(lambda x: stats.zscore(x))

        return self.X_df
    
    def preprocess(self):
        """
        Apply pre-processing to the x_data and y_data.
        input: all_fc_data = a df of the x_features, behav = a pandas series of behavioral scores
        output: a df of modified x_features according wanted params (fisher_z/scaled/both/none) and a df of modified y 
        """
        self.X_df = self._prepare_X()
        self.y_df = self._prepare_y()

    

    def get_x(self):
        return self.X_df.copy()

    def set_x(self, other_df):
        self.X_df = other_df.copy()

    def get_y(self):
        return self.y_df.copy()

    def set_y(self, other_df):
        self.y_df = other_df.copy()

    def split_train_test(self, dbg = False):
        """
        Splits the data into train and test datasets according to the test_size variable.
        Also shuffles the data.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                                            self.X_df, # x
                                                            self.y_df, # y
                                                            test_size = self.params_dict["test_size"], # e.g. 60%/40% split  
                                                            shuffle = True, # shuffle dataset
                                                                            # before splitting
                                                            random_state = 0 # same shuffle each
                                                                            # time
                                                                            )

        if dbg:
            # print the size of our training and test groups
            print('training:', len(self.X_train),
                'testing:', len(self.X_test))
            
            #Examine the distributions of y_train and y_test
            sns.distplot(self.y_train,label='train')
            sns.distplot(self.y_test,label='test')
            plt.legend()


    #Fit functions:
    def override_params_func(self, override_params):
        """
        Set parameters according to the requested model by adding the model parameters to params_dict.
        """
        if override_params == None:
            override_params = copy.deepcopy(self.params_dict)

        for k in self.params_dict.keys():
            if k in override_params.keys():
                continue
            override_params[k] = self.params_dict[k]

        return override_params

    def fit_model(self, select_features = True, override_params = None):
        #print("Liz")
        override_params = self.override_params_func(override_params)
        print(override_params.keys())
        #select features
        if select_features:
            importlib.reload(feature_selec_functions)
            mask = np.ones(len(self.X_df.columns))
            if override_params["params_selec"]["selection_method"] == "univariate": 
                mask, corr = feature_selec_functions.select_features_uni(self.X_train, self.y_train, override_params["params_selec"]["percent"],
                    override_params["params_selec"]["corr_type"], verbose=False)  
            elif override_params["params_selec"]["selection_method"] == "multivariate": #need to write it
                mask, corr = feature_selec_functions.select_features_multi(self.X_train, self.y_train, override_params) 
            
            #filter x_train according to the selected features
            X_train_selec = feature_selec_functions.arrange_selected_data(self.X_train, mask, override_params["params_selec"]["corr_type"]) 
            
        #optimized cv
        out_vals, best_param, best_err = self._optimized_cv(X_train_selec, self.y_train, override_params)

        #fit to all the train data
        final_model = best_param
        final_model.fit(X_train_selec, self.y_train)

        #predict on the x_test
        ##filter the x_test features according to the mask_dict that was chosen for the x_train data
        X_test_selec = feature_selec_functions.arrange_selected_data(self.X_test, mask, override_params["params_selec"]["corr_type"]) 

        yhat = final_model.predict(X_test_selec)
        if override_params["params_predict"]["eval_score"] == "mean_squared_error": #evluation score
            final_model_score = [final_model, mean_squared_error(self.y_test, yhat)]

        return final_model_score

    def _inner_fit_fun(self, X, y, override_params):
        fit_dict = override_params['params_fit']
        models = []
        if self.model_name == 'ElasticNet':
            l1 = override_params['params_fit']['l1_ratio']
            for a in override_params['params_fit']['alpha']:
                print(a)
                models.append(ElasticNet(l1_ratio=l1, alpha=a)) #models to fit
                models[-1].fit(X=X, y=y)

        return models

    def _inner_eval_fun(self, X, y, models, override_params):
        scores_and_params_list = []
        if self.model_name == 'ElasticNet':
            #fit_dict = override_params['params_fit']
            for m in models:
                #models.append(ElasticNet)
                print(m)
                yhat = m.predict(X=X) #predict yhat using x_eval
                print(yhat)
                #curr_params = {'l1_ratio': fit_dict["l1_ratio"][0], 'alpha': fit_dict["alpha"][i]}
                if override_params["params_predict"]["eval_score"] == "mean_squared_error": #evluation score
                    scores_and_params_list.append([m, mean_squared_error(y, yhat)])
        return scores_and_params_list

    def _optimized_cv(self, X, y, override_params, nfolds=2): 
        """
        Splits the data into nfolds datasets, fit the data to (nfolds-1)/nfolds of the data 
        and then evaluate on the 1/nfold of the data.
        Returns the best parameters according to the CV.
        """
        cv_id = np.arange(len(X))
        out_vals = []
        for i in range(nfolds):
            ids_train = cv_id % nfolds != i #not equal to i
            ids_eval = np.logical_not(ids_train) #take for evaluation the samples that were not used for training
            print("starting fold {}".format(i))

            mdls = self._inner_fit_fun(X.values[ids_train,:], y.values[ids_train], override_params)
            #print(mdls)
            out_vals.append(self._inner_eval_fun(X.values[ids_eval,:], y.values[ids_eval], mdls, override_params)) 
            #list of len n-folds of list of len num_param_values of list of len 2 (model+score)
            

        # take the out_vals and calculate the best options
        n_options = len(out_vals[0])
        best_err = np.Inf
        best_param = None
        for i in range(n_options):
            curr_option_err = 0 
            for j in range(nfolds):
                curr_option_err = out_vals[j][i][1]
                if curr_option_err < best_err:
                    best_err = curr_option_err
                    best_param = out_vals[0][i][0] #doesn't matter from which n_fold I take the params because they are the same
        
        #assert(not best_param) #validate that the best_params have been calculated = that they are not None

        return out_vals, best_param, best_err 



