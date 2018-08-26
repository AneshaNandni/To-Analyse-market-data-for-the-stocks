
import numpy as np
import sklearn
import scipy.stats
from pydoc import help
import sklearn.preprocessing as preprocessing

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

from sklearn.svm import SVR
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFE    
from sklearn.datasets import make_classification

from sklearn.linear_model import RandomizedLasso
 import pandas as pd
    df=pd.read_csv('C:/Users/NCS/Desktop/trainlabels.csv')
    df.head()


np.set_printoptions(threshold=100)
nb_datasets = 30


trainLabels = np.genfromtxt(
    fname = 'trainLabels.csv',
    skip_header = 1,
    delimiter = ',',
    )


data = {}
for x in range(1,nb_datasets+1):
    data[x] = np.genfromtxt(
    fname = 'data/%d.csv' %x,
    skip_header = 1,
    delimiter = ',',
    )
    

outputs = {}
for x in range(1,199):   
    outputs[x] = np.array(data[1][:,x-1]) 
    for y in range(2,nb_datasets):
        outputs[x] = np.append(outputs[x],data[y][:,x-1]) 

 
features = {}
for x in range(1,245):
    features[x] = np.array(data[1][:,x+197])
    for y in range(2,nb_datasets):
        features[x] = np.append(features[x],data[y][:,x+197])


inputs = {}
dataset = 1
y = 1
total_datapoints = nb_datasets*55  
inputs[x] = np.array(data[dataset][y-1][198:])
while( x < total_datapoints+1 and dataset <= nb_datasets):
    if(y < 55):       
        x = x + 1
        inputs[x] = np.array(data[dataset][y][198:]) 
        y = y + 1
    else:
        dataset = dataset+1  
        y = 0

def static_preprocess_features(features, type = 'standardize', indices = None):  fied
    if(type == 'normalize'):
        static_normalize_features(features, indices)
    else:
        static_standarize_features(features, indices)

def preprocess_features(features, type = 'standardize', indices = None):  
    if(type == 'normalize'):
        return normalize_features(features, indices)
    else:
        return standarize_features(features, indices)


def static_standarize_features(features, indices= None):
    if(indices): 
        for x in indices:
            std_scale = preprocessing.StandardScaler().fit(features[x])
            features[x] = std_scale.transform(features[x])
    else:        
        for x in range(0,len(features)):
            std_scale = preprocessing.StandardScaler().fit(features[x])
            features[x] = std_scale.transform(features[x])


def standarize_features(features, indices= None):
    std_features = {}
    if(indices):  
        for x in indices:
            std_scale = preprocessing.StandardScaler().fit(features[x])
            std_features[x] = np.array(std_scale.transform(features[x]))
    else:        
        for x in features:
            std_scale = preprocessing.StandardScaler().fit(features[x])
            std_features[x] = np.array(std_scale.transform(features[x]))
    return std_features


def normalize_features(features, indices = None):
    norm_features = {}
    if(indices):
        for x in indices:
            feat_min = min(features[x])
            feat_max = max(features[x])
            norm_features[x] = np.array([(feat_val-feat_min)/(feat_max-feat_min) for feat_val in features[x]])
    else:     
        for x in features:
            feat_min = min(features[x])
            feat_max = max(features[x])
            norm_features[x] = np.array([(feat_val-feat_min)/(feat_max-feat_min) for feat_val in features[x]])
    return norm_features
    
features = preprocess_features(features, 'normalize')

    

def get_correlation_matrix2(features, indices = None):     
    if(indices):
        corr = np.zeros([len(indices),len(indices)])   
        p_corr = np.zeros([len(indices),len(indices)])
        for x in indices:
            for y in indices:
                corr[x][y] = scipy.stats.pearsonr(features[x],features[y])[0]   
                p_corr[x][y] = scipy.stats.pearsonr(features[x],features[y])[1]
    else:
        corr = np.zeros([len(features),len(features)])   
        p_corr = np.zeros([len(features),len(features)]) 
        for x in range(1,len(features)):
            for y in range(1,len(features)):
                corr[x][y] = scipy.stats.pearsonr(features[x],features[y])[0]   .
                p_corr[x][y] = scipy.stats.pearsonr(features[x],features[y])[1]
    return (corr, p_corr)
    

def get_correlation_matrix(features, indices = None): 
    corr = {}   
    p_corr = {}
    if(indices):
        for x in indices:
            for y in indices:
                corr[(x,y)] = scipy.stats.pearsonr(features[x],features[y])[0]   
                p_corr[(x,y)] = scipy.stats.pearsonr(features[x],features[y])[1]
    else:
        for x in features:
            for y in features:
                corr[(x,y)] = scipy.stats.pearsonr(features[x],features[y])[0]   
                p_corr[(x,y)] = scipy.stats.pearsonr(features[x],features[y])[1]
    return (corr, p_corr)


(feat_corr, feat_p_corr) = get_correlation_matrix(features)



def filter_matrix_by_threshold(values, threshold, keep = 'greater'):
    filtered = []
    for x in range(0,values.shape[0]):
        for y in range(0,values.shape[1]):
            if(keep == 'greater'):
                if(values[x][y] >= threshold):
                    filtered.append((x,y))
            else:
                if(values[x][y] <= threshold):
                    filtered.append((x,y))
    return filtered
    

def filter_tuples_by_threshold(tuples, threshold, keep = 'greater'):
    filtered = {}
    for tuple in tuples:
        if(keep == 'greater'):
            if(tuples[tuple] >= threshold):
                filtered[tuple] = tuples[tuple]
        else:
            if(tuples[tuple] <= threshold):
                filtered[tuple] = tuples[tuple]
    return filtered


def filter_values_by_threshold(values, threshold, keep = 'greater'):
    filtered = {}
    for key in values:
        if(keep == 'greater'):
            if(values[key] >= threshold):
                filtered[key] = values[key]
        else:
            if(values[key] <= threshold):
                filtered[key] = values[key]
    return filtered

def filter_features_by_values(features, values):
    filtered_features = {}
    for key in values:
        if not (key in filtered_features):
            filtered_features[key] = features[key]
    return filtered_features


def find_variance(features):
    var = {}
    for x in features:
        var[x] = np.var(features[x])
    return var
    
feat_var = find_variance(features)


def generate_axis(start, end, step):
    axis = np.zeros(int(end+1/step))
    for i in range(start, end+1):
        axis[i] = i
    axis = axis[1:]
    return axis
    
feat_x_axis = generate_axis(0,len(feat_var),1)

def plot_key_value(d: dict):
    x_axis = np.fromiter(d.keys(), dtype = int)
    y_axis = [float(value) for value in d.values()]
            
    plt.plot(y_axis)
    plt.show()
    
plot_key_value(feat_var)



high_var_feats_indices = filter_values_by_threshold(feat_var, 0.05)


high_var_features = filter_features_by_values(features,high_var_feats_indices)


high_var_std_features = preprocess_features(high_var_features, 'standardize')


(high_var_feat_corr,high_var_feat_p_corr) = get_correlation_matrix(high_var_std_features)


final_tuples = filter_tuples_by_threshold(high_var_feat_p_corr,0.6)


final_feature_ids = [value[0] for value in final_tuples.keys()]
final_feats = filter_features_by_values(features,final_feature_ids)


final_inputs = {}
for x in range(0,len(outputs[1])):
    final_inputs[x] = np.zeros(len(final_feats))
    count = 0
    for key in final_feats:
        final_inputs[x][count] = final_feats[key][x]
        count = count+1

inputs = [input for input in final_inputs.values()]


svr = SVR(kernel="linear")
rfe = RFE(svr, step=1)
rfe = rfe.fit(inputs,outputs[1])
rfe.support_
rfe.ranking_



selected_features = []
count = 0
for key in final_feats.keys():
    if (rfe.support_[count] == True):
        selected_features.append(key)
    count = count + 1
    
    

rlasso = RandomizedLasso(alpha=1)
rlasso.fit(inputs, outputs[2])
rlasso.scores_
