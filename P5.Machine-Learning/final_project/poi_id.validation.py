#!/usr/bin/python

import sys
import numpy as np
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA

def mkRatio(a, b):
    if a == 'NaN':
        a = 0
    if b == 'NaN' or b == 0:
        return('NaN')
    return(a/(b*1.0))

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [] # You will need to use more features
ignore_list = ['email_address']#,'deferred_income','restricted_stock'
#'restricted_stock_deferred','deferral_payments']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
wordData = pickle.load(open("p5_analyzedEmailData.pkl", "r") )

for p in data_dict:
    for f in data_dict[p]:
        if f not in features_list and f not in ignore_list:
            features_list.append(f)

dataset = {}
# Cleanse negative values from the (non-negative) fields that are retained.
for p in data_dict:
    for f in features_list:
        if data_dict[p][f] < 0:
            data_dict[p][f] = 'NaN'
            
### Task 2: Remove outliers
data_dict.pop('TOTAL')

### Task 3: Create new feature(s)

# First we integrate the results of the tlidf word study
for p in data_dict:
    mailA = data_dict[p]['email_address'] 
    if mailA in wordData:
        data_dict[p]['flaggedMail'] = wordData[mailA]['flaggedMails']
    else:
        data_dict[p]['flaggedMail'] = 'NaN'
        
for p in data_dict:
    data_dict[p]['flaggedRatio'] = mkRatio(data_dict[p]['flaggedMail'],
            data_dict[p]['from_messages'])
    if 'flaggedRatio' not in features_list:
        features_list.append('flaggedRatio')   

    data_dict[p]['fromRatio'] = mkRatio(data_dict[p]['from_this_person_to_poi'],
            data_dict[p]['from_messages'])
    if 'fromRatio' not in features_list:
        features_list.append('fromRatio')    
    data_dict[p]['toRatio'] = mkRatio(data_dict[p]['from_poi_to_this_person'],
            data_dict[p]['to_messages'])
    if 'toRatio' not in features_list:
        features_list.append('toRatio')
    data_dict[p]['salToBonus'] = mkRatio(data_dict[p]['salary'],
            (data_dict[p]['bonus']))
    if 'salToBonus' not in features_list:
        features_list.append('salToBonus')

### Store to my_dataset for easy export below.
my_dataset = data_dict

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Extract features and labels from dataset for local testing

#features_list = ['poi','salary','bonus']
from sklearn.feature_selection import SelectKBest
skb = SelectKBest(k=5)
skb = skb.fit(features, labels)
features = skb.transform(features)
fs = skb.get_support()
