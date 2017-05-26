#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:58:59 2017

@author: max
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:10:56 2017

@author: max
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score #, f1_score
from sklearn.base import TransformerMixin, BaseEstimator
import xgboost as xgb

#os.chdir('/home/max/gridcure_domino/')
from src.gridcure_utilities import reshape_wide_to_long, scale_series, decompose, stats_by_house

############################################################################### 
# Wrap feature-generation/selection for predict-EV-by-house in a class w/
#
# fit/transform methods, for use with Scikit-Learn CV & model tuning tools
###############################################################################

class features_by_house(BaseEstimator, TransformerMixin):
    
    def __init__(self, 
                 signal_components = ['observed'],
                 return_type = 'numpy_array'):

        # Input validation
        assert return_type in ['numpy_array', 'pandas_dataframe']
        assert set(signal_components).issubset(['observed', 'resid', 'trend', 'seasonal'])
        
        # Assign required params as self-attributes so can be accessed by transform()
        self.return_type = return_type
        self.signal_components = signal_components
    
    def fit(self, X, y=None):
        return self #stateless transformations, do not peek at y
        
    def transform(self, 
                  X, 
                  y=None):
        
        '''Generate features from raw meter data using Pandas & Statsmodels
                
            Test at end to confirm reshaping and pandas processing has
            not changed order of rows (which correspond to order of houses)        
        
        -----------
        Parameters:
                X:          Pandas Dataframe of raw meter signal, like EV_train.csv. 
                            Shape (num_houses, num_intervals)
                
                y:          Series of labels for whether an EV ever charged at
                            a home. Shape (num_houses, )
                            
            Output shape:   (num_houses, num_features).
                
        '''
        # Define self attributes as local vars (makes code easier to step-through)
        signal_components = self.signal_components
        return_type       = self.return_type
        
        # Create vector of original House-ID order (to ensure final order is unchanged)
        # (reset_index used otherwise held-over row-index messes up comparison with final-house-order at end)
        original_house_order = X['House ID'].reset_index(drop=True)
                
        # Reshape X/Y Wide-to-Long, so I can use Pandas.groupby().apply()
        X_reshaped_long = reshape_wide_to_long(X, value_col_name='Meter Reading')    
    
        # Drop house/interval combos with blank values in X (otherwise sm decompose() throws error)
        X_reshaped_long = X_reshaped_long.dropna(subset=['Meter Reading'])
        assert X_reshaped_long.isnull().sum().sum()==0
    
        # group meter data by house before feature-engineering steps
        X_grouped = X_reshaped_long.groupby('House ID', sort=False)        
        
        # Replace "Meter Reading" Column with decomposed signal component(s) in [signal_components]
        for c in signal_components:
    
            X_reshaped_long[c] = X_grouped['Meter Reading'].transform(lambda s: decompose(s.values,
                                                                                          component=c,
                                                                                          freq=48))
        X_reshaped_long = X_reshaped_long.drop('Meter Reading', axis=1) 
        
        # Drop NaNs created by decompose() sliding-window calc (first/last 24 intvls now have blanks)
        X_reshaped_long = X_reshaped_long.dropna()        
        
        # Final features - standard stats (e.g. mean/std/skew) on selected meter signal components
        final_features = stats_by_house(X_reshaped_long, signal_components[0])
        
        for c in signal_components[1:]:
            
            # stats-by-house returns stats indexed on house-id, so joins happen implicitly on house-id 
            final_features = final_features.join(stats_by_house(X_reshaped_long,
                                                                    data_col = c))
    
        # Results validation:
        # Confirm rows/houses are still in same order as original X
        final_house_order = pd.Series(final_features.index)
        assert final_house_order.equals(original_house_order)
                                
        # Return features as pd dataframe or numpy ndarray (no col names)
        return final_features if return_type=='pandas_dataframe' else final_features.values
        
        

################################################
# Evaluate different models for predicting  
# EVs by House
################################################

if __name__ == '__main__':

    # Download raw meter data and labels.
    meter_readings    = pd.read_csv('EV_files/EV_train.csv')
    EV_charging_flags = pd.read_csv('EV_files/EV_train_labels.csv')

    # Get y from Y ((m-by-1) vector of 1/0 labels, for whether EV ever charged at House)
    y_by_home = EV_charging_flags.drop('House ID', axis=1).max(axis=1)

    # Split into train/test.
    # Even though gridsearchCV will report held-out-set accuracy, I want
    # to refit/predict gridsearch.best-estimator on a truly held out set
    # to sanity-check my reported accuracy
    
    X_train, X_test, y_train, y_test = train_test_split(meter_readings, 
                                                        y_by_home, 
                                                        test_size=0.33, 
                                                        random_state=42)
        
    # Define predictor as pipeline of feature-generation and xgb classifier
    clf = xgb.XGBClassifier()
    pred_house_steps = [('featureGeneration', features_by_house(signal_components)),
                        ('clf', clf)]
    pred_home_pipeline = Pipeline(pred_house_steps)

    # Tune Pipeline. Try different sets of decomposed-signal components
    # and different XGB params
    param_grid = {'featureGeneration__signal_components': [['observed'],
                                                      ['resid'],
                                                      ['resid', 'trend', 'seasonal']
                                                      ],
                  'clf__n_estimators': [250],
                  'clf__max_depth': [2,3,5],
                  'clf__min_child_weight': [2,5],
                  'clf__colsample_bytree': [0.4, 0.75, 1.0],
                  'clf__reg_alpha': [0,1,2,3], # (L1 penalty)
                  'clf__reg_lambda': [0,1],    # (L2 penalty)
                  } #should have also tried base_score 0.5 vs 0.3

    grid_search = GridSearchCV(pred_home_pipeline, 
                           param_grid,
                           scoring='accuracy',
                           n_jobs=-1,
                           cv = 5,
                           verbose=1)        
        
    t0 = time.time()
    grid_search.fit(X_train, y_train)
    print("done in %0.0fs seconds" % (time.time() - t0))        

    # Compare grid-search's best-estimator's reported accuracy w/ accuracy on held-out data 
    best_grid_estimator = grid_search.best_estimator_ # 0.911 reported test-set accuracy
    
    best_grid_estimator.fit(X_train, y_train) 
    holdout_preds = best_grid_estimator.predict(X_test)
    print(accuracy_score(y_test, holdout_preds))      # .907

    
    # Save gridsearchCV object, with train/test scores for all pipeline versions    
    filename = 'ev_home_gridsearch_1.pkl'
    filepath = os.path.join('Model Tuning Results', filename)
    
    with open(filepath, 'wb') as handle:
        pickle.dump(grid_search, handle, protocol=pickle.HIGHEST_PROTOCOL)
    


        
        
# USE PIPELINE TO FIT/PREDICT ON TRAIN/TEST SPLIT
#X_train, X_test, y_train, y_test = train_test_split(meter_readings, 
#                                                    y_by_home, 
#                                                    test_size=0.33, 
#                                                    random_state=42)

# Fit pipeline
#pred_home_pipeline.fit(X_train, y=y_train)

# Make predictions
#train_preds = pred_home_pipeline.predict(X_train)
#test_preds  = pred_home_pipeline.predict(X_test)

# Score predictions
#print("Train accuracy is " + str(accuracy_score(y_train, train_preds))) 
#print("Test accuracy is " + str(accuracy_score(y_test, test_preds)))
