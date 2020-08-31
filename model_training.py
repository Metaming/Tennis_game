# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 13:25:37 2020


@author: Mingkai Liu
"""
# catboost for classification

import numpy as np
import pandas as pd 
import scipy as sp

import seaborn as sns
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from itertools import product

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import decomposition

import catboost
from catboost import CatBoostClassifier, Pool
 
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.regularizers import l2



def pca_matrix(matrix = None, col_numfeature = None,n_components = 1, svd_solver = 'full'):
    
    """This function take the matrix and return the result after principle component analysis
    col_numfeature is the numerical feature that requires PCA
    n_components is the number of PCA component"""
    
    # convert dataframe to numpy array
    array_n = matrix[col_numfeature].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()

    # standardize the data before performaing dimensionality reduction using PCA

    array_n = StandardScaler().fit_transform(array_n)

    # perform PCA
    pca = decomposition.PCA(n_components=n_components, svd_solver = svd_solver)
    pca.fit(array_n)

    array_pca = pca.transform(array_n)

    print('number of feature before PCA:',array_n.shape)
    print('number of feature after PCA:',array_pca.shape)

    #print(pca.explained_variance_)
    
    
    return array_pca


def train_test_split (matrix = None, flag_pca = 0, n_components = None, year_start = None, year_train = None,year_val = None, year_end = None,
                     col_catfeature = None, col_numfeature = None, col_target = None, one_hot_catfeature = False):

    """This function take the matrix and split into train,validation and test set using time-series split
    flag_pca define if PCA is required, n_components is the number of PCA component;
    col_catfeature define the categorical features, which won't be included in the PCA;
    col_numfeature define the numerical features, which will be included in the PCA;
    One_hot_catfeature define if it is necesary to perform onehot encoding for the categorical feature
     """
    
    if flag_pca == 0:
        
        array_pca = matrix[col_numfeature].replace([np.inf, -np.inf], 0)
        
        
    elif flag_pca == 1:

        # perform pca
        array_pca = pca_matrix(matrix = matrix, col_numfeature = col_numfeature,n_components = n_components, svd_solver = 'full')
   
    
    if one_hot_catfeature == True:
        #convert categorical data to one-hot encoding
        ohe = OneHotEncoder(handle_unknown='ignore')    
        df_ohe = ohe.fit_transform(matrix[col_catfeature]).toarray()

        # combine numerical features and categorical features
        X_n = np.concatenate((df_ohe,array_pca),axis=1)

    else:
        # combine numerical features and categorical features
        X_n = np.concatenate((matrix[col_catfeature],array_pca),axis=1)


    # define the index for train, validation and test sets
    index_train = matrix[(matrix.year>=year_start) &(matrix.year<=year_train)].index
    index_val = matrix[(matrix.year>year_train) & (matrix.year<=year_val)].index
    index_test = matrix[(matrix.year>year_val) & (matrix.year<=year_end)].index

    # generate features and targets 
    X_train = np.nan_to_num(X_n[index_train,:])
    X_val = np.nan_to_num(X_n[index_val,:])
    X_test = np.nan_to_num(X_n[index_test,:])

    y_train = matrix[(matrix.year>=year_start) &(matrix.year<=year_train)][col_target].to_numpy()
    y_val = matrix[(matrix.year>year_train) & (matrix.year<=year_val)][col_target].to_numpy()
    y_test = matrix[(matrix.year>year_val) & (matrix.year<=year_end)][col_target].to_numpy()
    
    # pack the train, validation and test sets    
    train_data = [X_train, y_train] 
    val_data = [X_val, y_val]
    test_data = [X_test, y_test]
    
    
    return train_data,val_data,test_data,X_n


def build_cat(loss_function='Logloss',eval_metric = 'Accuracy',od_wait=15,random_seed=42):
    
    #from catboost import CatBoostClassifier

    """
    This function define the parameters for catboost training
    loss_function is the metric used to optimize the model, while eval_metric is the metric used for overfitting detection,
    od_wait is the number of interation to wait for overfitting stop, od_type is thetype of the overfitting detector to use.
    """
    catb = CatBoostClassifier(loss_function=loss_function,eval_metric = eval_metric,
                              od_wait=od_wait,od_type='Iter',random_seed=random_seed)
    
    return catb


def train_cat(model=None,grid_para=None,cv=None,verbose=False,train_data=None,val_data=None,test_data=None,cat_features_index=None,
              loss_function = 'Logloss',eval_metric = 'Accuracy',plot_progress = False,retrain_best = True):
    
    """
    This function takes the train, validation and test data from function 'train_test_split' parameters for catboost training.
    loss_function is the metric used to optimize the model, while eval_metric is the metric used for overfitting detection,
    od_wait is the number of interation to wait for overfitting stop, od_type is thetype of the overfitting detector to use.
    """
    
    X_train, y_train = train_data[0],train_data[1]
    X_val, y_val = val_data[0],val_data[1]
    X_test, y_test = test_data[0],test_data[1]
    
    #X_test, y_test = test_data[0],test_data[1]
    
    # get the column index of categorical features in the X_train
    #cat_features_index = np.where(np.in1d(X_train.columns.values,col_category))[0]

    """Instead of using the GridSearchCV for finding the best hyperparameters, we only use it for cross-validation,but keep
    all the results, including the hyperparameters and the metrics of the model in train, validation and test sets.
     """
    
    # pay attention, * is used before the list to unpack the list into multiple argument for function
    # https://stackoverflow.com/questions/36901/what-does-double-star-asterisk-and-star-asterisk-do-for-parameters
    
    cases = list(product(*list(grid_para.values())[0:]))  

    cases = pd.DataFrame(columns = grid_para.keys(), data = cases)
    
    for row in cases.itertuples():
        
        print(row)
        idx = row.Index
        
        """
        row_bracket generate a dictionary of the form {key_A: [value_A],key_B: [value_B], ...  }
        which is required for the GridSearchCV param input 
        """
        
        row_bracket = list(map(lambda x: [x], row[1:]))
        param = dict(zip(grid_para.keys(),row_bracket))
        print(param)

        catb_grid = GridSearchCV(model, param, cv = cv,n_jobs = 1,refit = True)    

        catb_grid.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                cat_features= cat_features_index,
                verbose=verbose          
                 )
     
        cases.loc[idx,'metric_cat_train'] = catb_grid.best_estimator_.get_best_score()['learn'][eval_metric]
        cases.loc[idx,'loss_cat_train'] =catb_grid.best_estimator_.get_best_score()['learn'][loss_function]
        cases.loc[idx,'metric_cat_val'] =catb_grid.best_estimator_.get_best_score()['validation'][eval_metric]
        cases.loc[idx,'loss_cat_val'] =catb_grid.best_estimator_.get_best_score()['validation'][loss_function]
        
        train_metric = catb_grid.best_estimator_.get_evals_result()['learn'][eval_metric]
        train_loss = catb_grid.best_estimator_.get_evals_result()['learn'][loss_function]
        val_metric = catb_grid.best_estimator_.get_evals_result()['validation'][eval_metric]
        val_loss = catb_grid.best_estimator_.get_evals_result()['validation'][loss_function]

    
        if plot_progress == True:

            plot_train_progress(param,loss_function,train_loss,val_loss,eval_metric,train_metric,val_metric)
            
            
    # retrain the best parameters
    
    
    best_index = cases.metric_cat_val.argmax()
    best_para = cases.loc[best_index,grid_para.keys()]
    row_bracket = list(map(lambda x: [x], best_para.to_dict().values()))
    param = dict(zip(grid_para.keys(),row_bracket))

    #print('best parameters:',param)
    
        
    if retrain_best == True:
        
        catb_grid = GridSearchCV(model, param, cv = cv, n_jobs = 1,refit = True)    

        catb_grid.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                cat_features= cat_features_index,
                verbose=verbose          
                 )

        train_metric = catb_grid.best_estimator_.get_evals_result()['learn'][eval_metric]
        train_loss = catb_grid.best_estimator_.get_evals_result()['learn'][loss_function]
        val_metric = catb_grid.best_estimator_.get_evals_result()['validation'][eval_metric]
        val_loss = catb_grid.best_estimator_.get_evals_result()['validation'][loss_function]

        plot_train_progress(param,loss_function,train_loss,val_loss,eval_metric,train_metric,val_metric)         

    return cases, best_para, catb_grid



def plot_train_progress(param,loss_function,train_loss,val_loss,eval_metric,train_metric,val_metric):
    
    """This function plot the trainning progress of the model,
    loss_function is the name of the loss function for training optimization
    train_loss and val_loss are the loss obtained for train and validation sets during the trainning
    eval_metric is the name of the evaludation metric for early stopping and model evaludation
    train_metric and val_metric are the evaludation metric for train and validation sets"""
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs',fontsize=15)
    ax1.set_ylabel(loss_function, color=color,fontsize=15)
    ax1.plot(train_loss, color=color,lw=2)
    ax1.plot(val_loss,'-.', color=color,lw=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(eval_metric, color=color,fontsize=15)  # we already handled the x-label with ax1
    ax2.plot(train_metric, color=color,lw=2)
    ax2.plot(val_metric, '-.',color=color,lw=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.legend(['train','validation'])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(param)
    plt.show()
    
    return


def eval_cat(catb_grid,col_catfeature,col_numfeature, thread_shold = 0, n_top = 10):
    
    """"This function takes the best estimator of the catboost model from function train_cat
    and plot the most important features, 
    thread_shold is the thread_shold for feature importance to filter the feature, by default, we set thread_shold = 0,
    n_top is the number of top features to show, by default n_top = 10"""
    
    
    features = col_catfeature +col_numfeature
    feature_importance = catb_grid.best_estimator_.feature_importances_
    
    featureimportance = pd.DataFrame()
    featureimportance['features'] = features
    featureimportance['feature_importance'] = feature_importance
    
    featureimportance.sort_values(['feature_importance'],ascending = False, inplace=True)
    
    plt.figure(figsize=(12,4))
    #sns.set(font_scale=1.0)
    sns.barplot(y ='feature_importance' ,x = 'features' ,data = featureimportance[featureimportance['feature_importance']>=thread_shold].head(n_top) )
    plt.xticks(rotation = 90)
    plt.xlabel('Features',fontsize = 12)
    plt.ylabel('Feature Importance',fontsize = 12)
    plt.grid()
    plt.show()