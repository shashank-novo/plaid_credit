#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Sarfraz Ahmned
# Created Date: 29/07/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Contains the modules for model evaluations, automl, hyperparameter tuning and feature importance"""
# ---------------------------------------------------------------------------

# core libraries
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, f1_score
from scipy.stats import ks_2samp
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV

# plotting libraries
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import scikitplot as skplt


# Model Evalution Metrics --------------------------------------------------------------------------------------------------------
def model_metrics(y_pred: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray):
    '''
    Returns the evaluation metrics for a binary classification model

        Parameters:
                y_pred (array(int)): array of predicted labels consisting of only 0 and 1
                y_true (array(int)): array of actual labels consisting of only 0 and 1
                y_prob (array(int)): 2 dimensional array of predicted probabilities
        
        Prints:
                accuracy score (int): accuracy score of the binary classifier
                confusion matrix (int): confusion matrix of the binary classifier
                fpr (int): false positive rate of the binary classifier
                tpr (int): true positive rate of the binary classifier
                auc (int): area under curve score for the binary classifier
                sensitivity (int): sensitivity of the binary classifier
                specificity (int): specificity of the binary classifier
                f1-score (int): F1-score of the binary classifier
                ks-score (int): KS score of the binary classifier
                classification report (int): elaborate precision recall report of the binary classifier
                gain chart (plot): gain chart plot of the binary classifier
                lift chart (plot): lift chart plot of the binary classifier
                
        Raises:
                ValueError: If length of y_pred is not equal to length of y_true or X_test
    '''

    # Check if arrays are of different length
    if len(y_pred) != len(y_true):
        raise ValueError('Length of y_pred and y_true is not matching')

    # Check if arrays are of different length
    if len(y_pred) != len(y_prob):
        raise ValueError('Length of y_pred and y_prob is not matching')

    # Accuracy Score
    print('Accuracy Score:', np.round(accuracy_score(y_true, y_pred), 2), '\n')

    # Confusion Matrix
    cmtx = pd.DataFrame(
    confusion_matrix(y_true, y_pred, labels=[0, 1]), 
    index=['true:0', 'true:1'], 
    columns=['pred:0', 'pred:1'])
    print('Confusion Matrix:')
    print(cmtx, '\n')

    # FPR, TPR, AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    print('False Positive Rate:', np.round(fpr[1],2))
    print('True Positive Rate:', np.round(tpr[1],2))
    print('AUC:', np.round(auc(fpr, tpr), 2), '\n')

    # Sensitivity, Specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print('Sensitivity:', np.round(tp/(tp+fn),2))
    print('Specificity:', np.round(tn/(tn+fp),2), '\n')

    # F1 Score
    print('F1 Score:', np.round(f1_score(y_true, y_pred), 2), '\n')

    # KS Statistic
    print('KS Score:')
    print(ks_2samp(y_pred, y_true), '\n')
    
    # Classification Report
    print('Classification Report:')
    print(classification_report(y_true, y_pred))

    # Gain chart
    print('Gain Chart')
    skplt.metrics.plot_cumulative_gain(y_true, y_prob, figsize=(10, 6), title_fontsize=18, text_fontsize=16)
    plt.show()

    # Lift chart
    print('Lift Chart')
    skplt.metrics.plot_lift_curve(y_true, y_prob, figsize=(10, 6), title_fontsize=18, text_fontsize=16)
    plt.show()


# Cross Validation --------------------------------------------------------------------------------------------------------
def cross_validation(model: object, X: np.ndarray, y: np.ndarray, scoring='accuracy', folds=5, seed=42):
    '''
    Returns the cross validation scores for a binary classification model

        Parameters:
                model (object): a binary classifier model object
                X (array(int)): multi-dimensional array of predictor variables
                y (array(int)): array of actual labels consisting of only 0 and 1
                scoring (str, optional): scoring method to be used, use sklearn.metrics.get_scorer_names() to see the complete list
                folds (int, optional): total splits of the data to get the scores
        
        Returns:
                scores (int): cross validated scores of the binary classifer
                
        Raises:
                ValueError: If length of X is not equal to length of y
    '''
    # Check if arrays are of different length
    if len(X) != len(y):
        raise ValueError('Length of predictor and dependent is not matching')

    # Define splits
    cv = KFold(n_splits=folds, random_state=seed, shuffle=True)

    # Calculate scores
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    # Return scores
    return scores


# Feature importance --------------------------------------------------------------------------------------------------------
def feature_importance(model: object, X: pd.DataFrame, imp_type='gain', show_plot=False):
    '''
    Returns the best binary classifier with tuned hyperparameter set

        Parameters:
                model (object): model object of the binary classifier
                X (dataframe(pandas)): pandas dataframe of predictor variables in train dataset
                imp_type (str): importance type to be plotted from the model, choose from ['gain', 'cover', 'weight', 
                                                                                           'total_gain', 'total_cover']
        
        Prints:
                feat_importances (plot): plots the feature importance

        Returns:
                feat_importance (dataframe(pandas)): pandas dataframe of feature importances

        Raises:
                ValueError: If X is not pandas dataframe
    '''
    # Check if train dataset is passed as dataframe
    if type(X) != pd.core.frame.DataFrame:
        raise ValueError('Train dataset not passed as pandas dataframe')
    
    # Check if the classifier is tree based
    if type(model).__name__ in (['DecisionTreeClassifier', 'RandomForestClassifier', 'XGBClassifier']):
        importance_array = np.fromiter(model.get_booster().get_score(importance_type=imp_type).values(), dtype=float)
    else:
        importance_array = model.coef_[0]
    
    # Extract the feature importance
    feat_importances = pd.DataFrame(importance_array, index=X.columns, columns=['importance'])

    # Plot the feature importance
    if show_plot:
        fig = px.bar(feat_importances, orientation='h')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=500, width=1400)
        fig.show()

    # Return the feature importance dataframe
    return feat_importances


# Probability binning --------------------------------------------------------------------------------------------------------
def probability_bins(model: object, X: pd.DataFrame, 
                     target_col: str, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], aggregate_func='median'):
    '''
    Returns the binned probability with aggregated target column

        Parameters:
                model (object): model object of the binary classifier
                X (dataframe(pandas)): pandas dataframe of predictor variables in train dataset
                target_col (str): name of the column to be aggregated
                aggregate_func (str): name of the aggregate function to apply

        Returns:
                prob_dist (dataframe(pandas)): pandas dataframe of feature importances

        Raises:
                ValueError: If X is not pandas dataframe
    '''
    # Check if train dataset is passed as dataframe
    if type(X) != pd.core.frame.DataFrame:
        raise ValueError('Dataset not passed as pandas dataframe')

    # Get the class probability         
    predicted_probas = model.predict_proba(X.drop(columns=[target_col]))
    X['proba'] = predicted_probas[:,1:].flatten()

    # Define probability bins
    X['binned'] = pd.cut(X['proba'], bins)

    # Apply the aggregate function on the target column
    if aggregate_func == 'median':
        prob_dist = X.groupby('binned', as_index=False)[target_col].median()
    elif aggregate_func == 'mean':
        prob_dist = X.groupby('binned', as_index=False)[target_col].mean()
    elif aggregate_func == 'rate':
        prob_dist = X.groupby('binned', as_index=False).apply(lambda x: np.round(
            (x[target_col].sum()/len(x))*100, 2)).rename(columns={None:target_col+'_rate (%)'})
    
    # Add volume info to the aggregate dataframe
    prob_dist['volume'] = X.groupby('binned', as_index=False)[target_col].count()[target_col]
    prob_dist['volume %'] = np.round((prob_dist['volume'] / prob_dist['volume'].sum())*100, 2)

    # Return the aggregated dataframe
    return prob_dist



