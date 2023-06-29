#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Sarfraz Ahmned
# Created Date: 13/10/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Contains the modules for model building, feature importance and hyperparameter tuning"""
# ---------------------------------------------------------------------------

# core libraries
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, f1_score
from scipy.stats import ks_2samp
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# plotting libraries
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
# import scikitplot as skplt


# Train Test Split --------------------------------------------------------------------------------------------------------
def split_test_train(df: pd.DataFrame, target_column: str, test_size: float = 0.30, random_state: int = None):
        '''
        Returns the train and test datasets of the recieved data

            Parameters:
                    data (dataframe(pandas)): pandas dataframe of independent variables
                    target_column (str): dependent variable present in the dataframe
                    test_size (int): fraction of data need to be separated for testing
                    random_state (int): seed for reproducible result
            Returns:
                    x_train (dataframe(pandas)): training dataset of independent variables
                    y_train (dataframe(pandas)): training dataset of dependent variable
                    x_test (dataframe(pandas)): test dataset of independent variables
                    y_test (dataframe(pandas)): test dataset of dependent variable
        '''

        X_train, X_test, y_train, y_test = train_test_split(df.drop(target_column, axis=1),
                                                            df[target_column],
                                                            test_size=test_size,
                                                            stratify=df[target_column], 
                                                            random_state=random_state)
        X_train = X_train
        X_test = X_test
        y_train = y_train
        y_test = y_test

        return X_train, y_train, X_test, y_test


# Hyperparameter tuning --------------------------------------------------------------------------------------------------------
def tune_hyperparameters(X: np.ndarray, y: np.ndarray, model: object, parameters: dict, metric='accuracy', cv_folds=5):
    '''
    Returns the model object with tuned hyperparameters

        Parameters:
                X (array(int)): multi-dimensional array of predictor variables
                y (array(int)): array of actual labels consisting of only 0 and 1
                model (object): model object of the binary classifier
                parameters (dict): dict of hyperparameters to be used for searching
                metric (str, optional): scoring method to be used, use sklearn.metrics.get_scorer_names() to see the complete list
                cv_folds (int, optional): total splits of the data to fit the model

        Returns:
                best_estimator (object): model object with tuned hyperparameters

        Raises:
                ValueError: If length of X is not equal to length of y
    '''
    # Check if arrays are of different length
    if len(X) != len(y):
        raise ValueError('Length of predictor and dependent is not matching')

    # Initialize the grid search object with the required params
    grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring = metric, n_jobs = -1, cv = cv_folds, verbose=False)

    # Fit and search the optimal hyperparameters
    grid_search.fit(X, y)

    # Return the best fit model object
    return grid_search.best_estimator_


# Feature Encoding --------------------------------------------------------------------------------------------------------
def feature_encoding(data=pd.DataFrame(), encode_columns=[], encode_type='onehot', max_unique_values=20):
    """
    Function to encode the categorical variables
    'data' is necessary parameter and 'encode_columns' & 'encode_type' are optional parameters

        Parameters:
            data (dataframe): Dataframe dataset
            encode_columns (list): List of columns that require encoding
            encode_type (string): 'onehot' or 'label' encoding methiods

        Returns:
            data (dataframe): Transformed dataframe

        Raises:
            TypeError: If lenght of input data is zero
    """     
    if data.shape[0] > 0:

        # get columns from user or by data types
        if len(encode_columns) == 0:
            cat_columns = [col for col in data.columns if data[col].dtype.name in ['object','category','bool']]
        else:
            cat_columns = encode_columns

        # filter out columns that are unique identifiers
        cat_columns = [col for col in cat_columns if data[col].agg(['nunique'])[0] <= max_unique_values]
        rest_columns = list(set(data.columns)-set(cat_columns))

        # encode with one-hot encoding
        if encode_type == 'onehot':
            cat_data = pd.get_dummies(data[cat_columns])
            if len(rest_columns) > 0:
                rest_data = data[rest_columns]
                data = pd.concat([rest_data, cat_data], axis=1)
            else:
                data = cat_data

        # encode with label encoding
        elif encode_type == 'label':
            data_tmp = pd.DataFrame(columns=cat_columns)
            for col in cat_columns:
                data_tmp[col] = data[col].astype('category').cat.codes

            if len(rest_columns) > 0:
                rest_data = data[rest_columns]
                data = pd.concat([rest_data, data_tmp], axis=1)
            else:
                data = data_tmp
        else:
            raise ValueError('Invalid encoding type')
    else:
        raise TypeError('No data input or input data has zero records')
    return data


# Model Building --------------------------------------------------------------------------------------------------------
def classification_models(x_train, y_train, params_log_reg={}, params_dtc={}, models=[]):
    """
    Function to train the linear, logistic, decision trees.
    'train_data' is necessary parameter and remaining are optional parameters
        Parameters:
            x_train (dataframe): Dataframe dataset
            y_train (dataframe): Dataframe dataset
            params_log_reg (dict): logistic regression parameters
            params_dtc (dict): decision tree parameters
            models (list): ['log_reg','svc','dtc','rfc','xgbc']
        Returns:
            log_reg (object): trained model output
            dtc (object): trained model output
    """
    if models == [] or 'log_reg' in models:
        if params_log_reg == {}:
            log_reg = LogisticRegression().fit(x_train, y_train)
        else:
            log_reg = LogisticRegression()
            log_reg.set_params(**params_log_reg)
            log_reg.fit(x_train, y_train)
        return log_reg
    if models == [] or 'dtc' in models:
        if params_dtc == {}:
            dtc = DecisionTreeClassifier().fit(x_train, y_train)
        else:
            dtc = DecisionTreeClassifier()
            dtc.set_params(**params_log_reg)
            dtc.fit(x_train, y_train)
        return dtc
    return None