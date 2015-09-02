import sys
import numpy as np
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

def mkRatio(a, b):
    if a == 'NaN':
        a = 0
    if b == 'NaN' or b == 0:
        return('NaN')
    return(a/(b*1.0))
    
### Task 1: Select what features you'll use.
features_list = ['poi'] 

# Ignore email address since it's text
ignore_list = ['email_address'] 

# Load the dictionary containing the dataset and my processed text data
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
wordData = pickle.load(open("p5_analyzedEmailData.pkl", "r") )

# Populate all the features to the Features list to start
for p in data_dict:
    for f in data_dict[p]:
        if f not in features_list and f not in ignore_list:
            features_list.append(f)
            
print(features_list)

### Task 2: Remove outliers
data_dict.pop('TOTAL')

### Task 3: Create new feature(s)
# First merge in the email text data we parsed in MailParser.py
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
        
# Normalize to facilitate Feature Selection and/or models where normalization 
# is important.
fts = {}
for f in features_list:
    if f not in ignore_list and f != 'poi':
        fts[f] = {'min':0,'max':0}
        for p in data_dict:
            if data_dict[p][f] < fts[f]['min'] and data_dict[p][f] != 'NaN':
                fts[f]['min'] =  data_dict[p][f]
            if data_dict[p][f] > fts[f]['max'] and data_dict[p][f] != 'NaN':
                fts[f]['max'] =  data_dict[p][f]

for f in features_list:
    if f not in ignore_list and f != 'poi':
        for person in data_dict:
            if data_dict[person][f] != 'NaN':
                data_dict[person][f] = data_dict[person][f] / (fts[f]['max']
                    -fts[f]['min'])
                
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Feature Selection
from sklearn.feature_selection import SelectKBest
skb = SelectKBest(k=2)
skb = skb.fit(features, labels)
features = skb.transform(features)
np.shape(features)
sup = skb.get_support()
# temp feature list
fl = ['poi']
print('Retained Features:')
for f in range(0,len(features_list)-1):
    if sup[f]:
        print(features_list[f+1], skb.scores_[f])
        fl.append(features_list[f+1])
        
features_list = fl
# Set up the cross validateion, run the selected model and print results.

validation_scores = {}
counter = 0

from sklearn import grid_search
from sklearn.tree import DecisionTreeClassifier

parameters = {'criterion':['entropy', 'gini'], 
              'max_depth':range(1, 4), 
              'min_samples_leaf':range(1, 20)}

clf = DecisionTreeClassifier()

gvc = grid_search.GridSearchCV(clf, parameters)
gvc.fit(features, labels)
print(gvc.best_params_)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=1, 
                             min_samples_leaf=5)
    
clf.fit(features, labels)


dump_classifier_and_data(clf, data_dict, features_list)
# Now print the Udacity test results
test_classifier(clf, data_dict, features_list)