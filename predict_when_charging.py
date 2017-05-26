#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 00:52:30 2017

@author: max
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, GroupKFold


#os.chdir('/home/max/gridcure_domino/')
from src.predict_EV_homes import features_by_house # not used but needed to unpickle gridsearch results
from src.gridcure_utilities import reshape_wide_to_long, scale_series, decompose, dataframe_imputer, stats_by_house

#######################################
# FEATURE-GENERATION FUNCTIONS FOR
# MODIFYING X-DATAFRAMES
#######################################
def get_intvl_encodings(X_df, y):
    ''' Learn mapping of Interval to:
            1) Mean(y)   per Interval
            2) St Dev(y) per Interval
    '''    
    # Get X & y in single DF, grouped on Interval    
    grouped_df = pd.concat([X_df, y], axis=1).groupby('Interval', sort=False)

    # Calc interval-encoded stats
    mean_charge_by_intvl = grouped_df['EV Charging Flag'].apply(np.mean) 
    stdv_charge_by_intvl = grouped_df['EV Charging Flag'].apply(np.std)     

    # Return Pandas Dataframe with both calculated series as columns
    # (reset_index() so 'Interval' is a col of the DataFrame, for simpler merging in .transform())
    return pd.DataFrame({'mean_charge_by_intvl': mean_charge_by_intvl,
                         'stdv_charge_by_intvl': stdv_charge_by_intvl}).reset_index()
    

def add_delta_features(df, input_colname='Meter Reading'):
    '''Add feature columns based on deltas from prev meter readings, 
        and grouped summary stats'''
        
    # Get original row-id order (house/interval) to ensure I don't change it 
    orig_order = df[['House ID', 'Interval']]        
        
    # Group by House ID for summary stats by group and accurate shift() calculations
    df_grouped = df.groupby('House ID', sort=False)
    
    # Calculate deltas for meter readings over diff time-frames
    df[input_colname + ' Delta Hours 0.5'] = df[input_colname] - df_grouped[input_colname].shift(1)
    df[input_colname + ' Delta Hours 1']   = df[input_colname] - df_grouped[input_colname].shift(2)
    df[input_colname + ' Delta Hours 1.5'] = df[input_colname] - df_grouped[input_colname].shift(3)
    df[input_colname + ' Delta Hours 2']   = df[input_colname] - df_grouped[input_colname].shift(4)
    df[input_colname + ' Delta Hours 12']  = df[input_colname] - df_grouped[input_colname].shift(24)
    df[input_colname + ' Delta Hours 24']  = df[input_colname] - df_grouped[input_colname].shift(48)
    
    # Calculate Mean and Std of Meter Readings and all Deltas, by House ID
    group_stat_cols = [input_colname + suffix for suffix in 
                       ['', ' Delta Hours 0.5', ' Delta Hours 1', 
                       ' Delta Hours 1.5', ' Delta Hours 2', ' Delta Hours 12', 
                       ' Delta Hours 24']]
    
    group_means = df_grouped[group_stat_cols].mean().add_prefix('Mean ').reset_index()
    group_st_devs=df_grouped[group_stat_cols].std().add_prefix('StDev ').reset_index()
                     
    # Join New Group Features into original dataframe
    final_df = df.merge(group_means,   how='left', on='House ID').merge(
                        group_st_devs, how='left', on='House ID')
    
    # Post-conditions. This fn() should not have changed num rows or row-order
    assert df.shape[0] == final_df.shape[0]
    assert final_df[['House ID', 'Interval']].equals(orig_order)

    return final_df
        



#######################################
# SCIKIT-LEARN CLASS WRAPPERS FOR 
# FEATURE-GENERATION FUNCTIONS ABOVE
#######################################
class add_interval_encodings(TransformerMixin, BaseEstimator):
    
    def __init__(self):
        pass
    
    def fit(self, X_df, y):
        '''Learn encoded features - mean(y) and std(y) by Interval'''
        self.new_columns = get_intvl_encodings(X_df, y)
        return self
    
    def transform(self, X_df):
        '''Merge learned interval-encodings into X_df'''
        return X_df.merge(self.new_columns, how='left', on='Interval')


class delta_features_by_signal_component(TransformerMixin, BaseEstimator):
    
    def __init__(self, signal_components):
        self.signal_components = signal_components
    
    def fit(self, X_df):
        return self #(stateless, does not peek at y)
    
    def transform(self, X_df):
        ''' Update X_df to add delta stats for each signal component'''
        
        signal_components = self.signal_components 
        
        # Replace 'Meter Signal' column with decomposed signal components
        mr_grouped = X_df.groupby('House ID', sort=False)['Meter Reading']

        for c in signal_components:    
            X_df[c] = mr_grouped.transform(lambda s: decompose(s.values, component=c, freq=48))
        X_df = X_df.drop('Meter Reading', axis=1) 
        
        # Add 'delta' features to X_df, for each signal component
        timedelta_features =  add_delta_features(X_df, input_colname=signal_components[0])
        for c in signal_components[1:]:
   
            # stats-by-house returns stats indexed on house-id, so joins happen implicitly on house-id 
            timedelta_features = add_delta_features.join(stats_by_house(X_df, data_col = c))
        
        return timedelta_features    




########################
# FINAL MODEL AS SKLEARN CLASS 
# W/ FIT() AND TRANSFORM()
########################

class features_by_house_by_interval(TransformerMixin, BaseEstimator):
    ''' Create feature matrix for predicting ev-charge by house by interval.
    
        The sklearn classes I use here were going to be wrapped in an Sklearn
        pipeline directly, but then only the numpy values (not pandas dataframe)
        are passed along each step. I use Pandas for data wrangling and want
        to pass an dataframe through feature-generation steps. Calling my 
        class fit/transform() methods in sequence here, in a 'feature gen' class,
        allows me to do that.
    
        One step (interval-encoder) is supervised and must be fit, 
        decomposition/delta features is not
    '''

    def __init__(self, signal_components = ['observed']):
        ''' Instantiate steps in my pipeline as attributes of self'''

        # Assert only valid signal components for statsmodels decompose() were passed. 
        # (time-delta features will be calculated on each signal component)
        assert set(signal_components).issubset(['observed', 'resid', 'trend', 'seasonal'])
        self.signal_components = signal_components
        
        # Instantiate steps for feature generation and classification (used in fit/transform)
        self.interval_encoder = add_interval_encodings()
        self.delta_features = delta_features_by_signal_component(signal_components)

        
    def fit(self, X, y):
        ''' Fit any supervised steps in my feature generation (just one for now)'''
                        
        # Learn mean(y) and stdv(y) by Interval, for X,y
        self.interval_encodings = self.interval_encoder.fit(X, y)
        
        # this fit just returns self, does nothing except make the .transform() work later
        self.delta_features.fit(X)

        return self

    
    def transform(self, X):
        ''' Add features to X
        '''
        
        # Add columns for mean(y) and stdv(y) by Interval to X, learned in fit() 
        X_2 = self.interval_encoder.transform(X)
        
        # Add 'delta' features, for each decomposed signal component in self.signal_components
        X_3 = self.delta_features.transform(X_2)
        
        # Drop ID columns which are not features for clf, and return feature-matrix
        feature_matrix = X_3.drop(['House ID', 'Interval'], axis=1)
        print('Created feature matrix, about to run xgb at ', time.strftime('%l:%M%p %Z on %b %d, %Y'))

        return feature_matrix
    
        

def main(cv_results_filename = '',
         run_or_load_cv = 'load',
         predictions_folder = '',
         final_predictions = 0):

    ''' Make Train and Test Predictions based on Grid-Searched Results
    
        params
        -------
        run_or_load_cv:     whether to run gridsearch and save, or whether to 
                            load prev results
                        
        final_predictions:  Whether to predict on EV_test, or divide labeled
                            EV_train into train/test sets for confirming 
                            performance generalizes
                            
        cv_results_filename: pickle filename to save or load CV object from
    
    '''

    # Input validation
    assert run_or_load_cv in ('run', 'load')
    assert final_predictions in (0,1)


    # Get X, Y  /  separate to train,test.
    if final_predictions == 1:
        X_train = pd.read_csv('EV_files/EV_train.csv')
        X_test  = pd.read_csv('EV_files/EV_test.csv')
        Y_train = pd.read_csv('EV_files/EV_train_labels.csv')
    
    else:        
        meter_readings    = pd.read_csv('EV_files/EV_train.csv')
        EV_charging_flags = pd.read_csv('EV_files/EV_train_labels.csv')
            
        X_train, X_test, Y_train, Y_test = train_test_split(meter_readings, 
                                                        EV_charging_flags, 
                                                        test_size=0.2, 
                                                        random_state=42)
    
    # Impute missing values in train/test meter-readings, otherwise statsmodels decompose() throws error
    X_train = dataframe_imputer(X_train, strategy='mean', axis=1)
    X_test  = dataframe_imputer(X_test,  strategy='mean', axis=1)


    ############################3#######
    # ADD PREDICTION OF <EV-AT-HOUSE> 
    # TO X_TRAIN/X_TEST DATA
    #####################################

    # Get true EV-status by house, to train clasifier for predicting which homes have EVs
    true_EV_by_house_train = Y_train.drop('House ID', axis=1).max(axis=1)

    # Load results of grid-search through predict-EV-by-house models, fit best model on train
    pred_house_cv_filepath = 'Model Tuning Results/ev_home_gridsearch_1.pkl'
    with open(pred_house_cv_filepath, 'rb') as handle:
        pred_house_cv = pickle.load(handle) 

    pred_EV_by_house = pred_house_cv.best_estimator_            
    pred_EV_by_house.fit(X_train, true_EV_by_house_train)

    # Add Predictions-by-House as Feature. 
    X_train['Predicted EV by House'] = pred_EV_by_house.predict(X_train)
    X_test[ 'Predicted EV by House'] = pred_EV_by_house.predict(X_test)
    
    # Export Predictions by House
    if final_predictions == 1:
        X_test[['House ID', 'Predicted EV by House']].to_csv(os.path.join('predictions', predictions_folder, 'test_preds_by_house.csv'), index=False)
    
        
    # Reshape X & Y so that each row has one House/Interval observation.
    # (This is the shape required by my 2nd level model, predicting by house by interval)
    X_train_reshaped = reshape_wide_to_long(X_train, value_col_name = 'Meter Reading')
    X_test_reshaped  = reshape_wide_to_long(X_test,  value_col_name = 'Meter Reading')

    Y_train_reshaped = reshape_wide_to_long(Y_train, value_col_name = 'EV Charging Flag')
    
 
    ##################################
    # RUN CV COMPARISON OF MODELS, 
    # (OR LOAD PREVIOSU RESULTS
    ##################################
    if run_or_load_cv == 'run':
        
        # Define pipeline components
        clf = xgb.XGBClassifier()
        
        pred_steps = [('featureGeneration', features_by_house_by_interval()),#signal_components)),
                      ('clf', clf)]
        pipeline = Pipeline(pred_steps)
        
        param_grid = {'featureGeneration__signal_components': [['observed'],
                                                               ['resid', 'seasonal'],
                                                          ['resid'],
                                                          ['resid', 'trend', 'seasonal'],
                                                          ],
                      'clf__n_estimators': [100],
                      'clf__max_depth': [3,5],
                      'clf__reg_alpha': [1,2,3], #(L1 penalty)
                      'clf__base_score': [0.03]
                      }
    
        # Group Kfold used so that training/testing always done on separate houses
        gkf = list(GroupKFold(n_splits=3).split(X_train_reshaped, 
                                           Y_train_reshaped['EV Charging Flag'],
                                           groups=X_train_reshaped['House ID']))
        
        # Define final CV space to search 
        cv_search = GridSearchCV(pipeline, 
                                   param_grid, 
                                   scoring='f1',
                                   cv = gkf,
                                   refit=True,
                                   verbose = 3,
                                   n_jobs=1) # xgb clf already uses all cores by default. 

        # Fit/Score all Models in Search Space, Save to disk
        cv_search.fit(X_train_reshaped, Y_train_reshaped['EV Charging Flag'])                                     

        filepath = os.path.join('Model Tuning Results', cv_results_filename)            
        with open(filepath, 'wb') as handle:
            pickle.dump(cv_search, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # Skip above and just load previous CV results if caller chose            
    elif run_or_load_cv == 'load': 
        with open(os.path.join('Model Tuning Results', cv_results_filename), 'rb') as handle:
            cv_search = pickle.load(handle) 
                                  
        

    ##################################
    # CV SEARCH COMPLETED OR LOADED
    # PROCEED TO FIT AND PREDICT
    ##################################                           
    best_model = cv_search.best_estimator_ # already fitted on all train data with refit=True
    
    #best_model.fit(X_train_reshaped, Y_train_reshaped['EV Charging Flag'], clf__tree_method='exact')

    train_preds = best_model.predict_proba(X_train_reshaped)
    test_preds  = best_model.predict_proba(X_test_reshaped)

    # Predict_Proba outputs 2D array. I want just probability of class=1
    index_of_class_1 = best_model.classes_.tolist().index(1)

    train_preds = train_preds[:,index_of_class_1]
    test_preds  = test_preds[:, index_of_class_1]


    # Add House/Interval Information back to Predictions Arrays
    train_preds_df = pd.concat([X_train_reshaped[['House ID', 'Interval']], pd.Series(train_preds)], axis=1)
    test_preds_df  = pd.concat([X_test_reshaped[ ['House ID', 'Interval']], pd.Series(test_preds)],  axis=1)

    train_preds_df.columns = ['House ID', 'Interval', 'EV Charging Flag']
    test_preds_df.columns  = ['House ID', 'Interval', 'EV Charging Flag']
    
    train_preds_df['House ID'] = train_preds_df['House ID'].astype(int)
    test_preds_df[ 'House ID'] = test_preds_df[ 'House ID'].astype(int)
    
    # Save train/test preds-df to disk in current shape for prediction performance report
    preds_path = os.path.join('predictions', predictions_folder)
    
    train_preds_df.to_csv(os.path.join(preds_path, 'train_preds.csv'), index=False)
    test_preds_df.to_csv( os.path.join(preds_path, 'test_preds.csv'),  index=False)

    # Save True Train-Labels for Prediction Performance Report:
    Y_train_reshaped.to_csv(os.path.join(preds_path, 'train_Y.csv'), index=False)

    # If test_preds are on held-out samples from EV_train.csv that we have labels for,
    # then save them for evaluating 'test set performance' in pred perf report
    if final_predictions == 0:
        Y_test_reshaped = reshape_wide_to_long(Y_test, value_col_name = 'EV Charging Flag')
        Y_test_reshaped.to_csv(os.path.join(preds_path, 'test_Y.csv'), index=False)

    # Feature Importances from XGB Clf
    d = best_model.steps[1][1].booster().get_fscore()

    feature_importances = pd.DataFrame(d, index=[0]).T.sort_values(by=0, ascending=False).reset_index()
    feature_importances.columns = ['Feature', 'XGB Reported Importance']
    
    feature_importances.to_csv(os.path.join(preds_path, 'feature_importances.csv'), index=False)
    
    
    # IF final_predictions == 1 and so test_preds are on actual unlabeled EV_test.csv,
    # Then reshape to sample_submission.csv shape (houses as rows)
    if final_predictions == 1:
        
        # Reshape test-preds-df to shape of sample_submission.csv 
        test_preds_submission = test_preds_df.set_index(['House ID', 'Interval']).unstack('Interval')
    
        test_preds_submission.columns = test_preds_submission.columns.droplevel()
        test_preds_submission.columns = ['Interval_'+str(x) for x in test_preds_submission.columns]
                    
        
        # Re-sort so that rows/houses are in same order as original ev_test.csv
        test_preds_submission = test_preds_submission.reset_index()

        row_order = X_test['House ID']
        test_preds_submission['House ID'] = pd.Categorical(test_preds_submission['House ID'], row_order)
        test_preds_submission.sort_values(by='House ID', inplace=True)

        test_preds_submission.to_csv(os.path.join(preds_path, 'test_preds_submission.csv'), index=False)




                
if __name__ == '__main__':

    main(cv_results_filename = 'gridsearch_when_charging_3_f1.pkl',
         run_or_load_cv = 'load',
         predictions_folder = 'preds_9_stacked_final_pred_on_EV_test_csv',
         final_predictions = 1)

    