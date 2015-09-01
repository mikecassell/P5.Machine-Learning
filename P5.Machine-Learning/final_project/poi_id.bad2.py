#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import pandas as pd
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.decomposition import PCA

def mkRatio(a, b):
    if a == 'NaN':
        a = 0
    if b == 'NaN' or b == 0:
        return('NaN')
    return(a/(b*1.0))
## Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features
features_list = ['salary','bonus','deferral_payments','exercised_stock_options']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))
my_dataset = data_dict
# Iterate through the possible fields so we have them all for exploration:
for person in my_dataset:
    for field in my_dataset[person]:
        if field not in features_list and field != 'email_address':
            features_list.append(field)

### Task 2: Remove outliers
data_dict.pop('TOTAL')
### Task 3: Create new feature(s)

#for r in data_dict:
#    data_dict[r]['fromRatio'] = mkRatio(data_dict[r]['from_this_person_to_poi'],
#            data_dict[r]['from_messages'])
#    features_list.append('fromRatio')    
#    data_dict[r]['toRatio'] = mkRatio(data_dict[r]['from_poi_to_this_person'],
#            data_dict[r]['to_messages'])
#    features_list.append('toRatio')
### Store to my_dataset for easy export below.

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

clf = svm.SVC()
clf.fit(features,labels)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)