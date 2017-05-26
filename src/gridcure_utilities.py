#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:31:58 2017

@author: max
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import Imputer

#def reshape_wide_to_long(df, value_col_name=''):
#    '''Convert EV_train/labels from matrix shape to long column of values, 
#        with ID/Interval as additional cols for groupby()
#    '''
    
    # Reshape from Wide-to-Long. 
    # (From 2880 Interval Cols to 2880 Interval Rows per house)
#    reshaped = pd.melt(df, 
#                       id_vars=['House ID'], 
#                       var_name='Interval', 
#                       value_name=value_col_name)
    
    #Convert Time 'Interval' Values to e.g. '1' instead of 'Interval_1' 
 #   reshaped['Interval'] = reshaped['Interval'].str[9:].astype(int)        
    
#   return reshaped    

#df = X_train
#value_col_name = 'Meter Reading'
def reshape_wide_to_long(df, value_col_name):

    ''' Reshape so that Interval is one col w/ 2880 levels, not 2880 cols

    (so I can use Pandas' df.groupby() for clearer data-wrangling)    
    '''
    #print('About to reshape_wide_to_long df with value-col ', value_col_name)
    #print(df.head(5))
    
    # Identify col names to keep as row identifiers - anything that's not 'Interval_'
    # (this will be just 'House ID' for Y_train, and 'House ID'/'Pred by House' for Xs)
    row_ID_vars = [col for col in df.columns if not col.startswith('Interval_')]    
    
    # reshape wide to long
    reshaped = pd.melt(df, 
                       id_vars = row_ID_vars,
                       var_name = 'Interval', 
                       value_name=value_col_name)

    #Convert 'Time Interval' Values to e.g. '1' instead of 'Interval_1' 
    reshaped['Interval'] = reshaped['Interval'].str[9:].astype(int)        

    return reshaped    


def dataframe_imputer(DF, strategy='mean', axis=1):
    ''' thanks O.rka http://stackoverflow.com/a/33661042/1870832 '''
    
    fill_NaN = Imputer(missing_values=np.nan, strategy=strategy, axis=axis)
    
    imputed_DF = pd.DataFrame(fill_NaN.fit_transform(DF))
    imputed_DF.columns = DF.columns
    imputed_DF.index = DF.index

    return imputed_DF

    
def scale_series(s):
    '''Scale Series to Achieve Mean=0, St. Dev=1
       Helper function to use with Pandas .apply/transform() 
       '''
    
    return (s - np.mean(s)) / np.std(s)

    
def decompose(s, component, freq=48):
    ''' Decompose signal into seasonality/trend/residual and return 
        component requested by user.
        
        Helper function to use with Pandas .apply/transform() 
    '''    
    
    # input validation
    assert component in ('seasonal', 'trend', 'resid', 'observed')
    
    # Create dictionary with each signal component sm returns as obj attributes
    decomposed = sm.tsa.seasonal_decompose(s, freq=freq)
    components_dict = {'seasonal': decomposed.seasonal,
                       'trend'   : decomposed.trend,
                       'resid'   : decomposed.resid,
                       'observed': decomposed.observed}
                       
    return components_dict[component]

    
def stats_by_house(df, 
                   data_col='Meter Reading', 
                   key_intvls = [37,38,39,40, 85,86,87,88,89, 133,134,135,136, 421,422,423,424,425, 469,470,471,472]):
        
    ''' Calculate stats of column [data_col], grouped by 'House ID'
    
        Key-intervals selected based on 
    '''
     
    grpd = df.groupby('House ID', sort=False)[data_col]

    # Calculate standard Series stats by HouseID
    features = pd.DataFrame()
    features[data_col+ ' max']      =   grpd.apply(np.max)
    features[data_col+ ' min']      =   grpd.apply(np.min) 
    features[data_col+ ' mean']     =   grpd.apply(np.mean)
    features[data_col+ ' median']   =   grpd.apply(np.median)    
    features[data_col+  ' St Dev']  =   grpd.apply(np.std)
    features[data_col+ ' skew']     =   grpd.apply(pd.Series.skew)
    features[data_col+ ' kurtosis'] =   grpd.apply(pd.Series.kurtosis) 
                
    # EV-homes tend to be charging 3-10% of intervals. So hopefully these
    # percentiles are helpful for distinguishing
    features[data_col+  ' 98th Percentile'] = grpd.apply(np.percentile, q=0.98)
    features[data_col+  ' 85th Percentile'] = grpd.apply(np.percentile, q=0.85)
    features[data_col+ ' Percentile Range'] = (features[data_col+ ' 98th Percentile'] - 
                                              features[data_col + ' 85th Percentile'])        
    # Avg during 'Key Intervals'
    # During these intervals, EVs tend to be charging (see "true_EV_by_intvl_notes on spikes.ods"),
    # and Meter Signals appear higher at EV homes than non-EV homes (see "Visualizing Meter Signals of EV- and non-EV-Homes" notebook)
    df_key_intervals = df[df['Interval'].isin(key_intvls)]
    features[data_col + ' mean at key intvls'] = df_key_intervals.groupby('House ID', sort=False)[data_col].apply(np.mean)
                          
    return features
