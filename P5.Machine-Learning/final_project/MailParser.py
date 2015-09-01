# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 08:17:19 2015
@author: mike
"""

#!/usr/bin/python

import pickle
import sys
import glob
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation as cv
from sklearn.feature_selection import SelectPercentile, f_classif
mypath = '/home/mike/Desktop/P5.Machine-Learning/final_project/'
mailDir = '/home/mike/projects/5. Machine Learning/'
mailListPath = '/home/mike/Desktop/P5.Machine-Learning/emails_by_address/'
sys.path.append( '../tools/')

from parse_out_email_text import parseOutText

data_dict = pickle.load(open("final_project_dataset.pkl", "r"))

"""
Helper functions
"""
# trimName(address)
# Takes an email file and trims the from_ and .txt to leave just the actual
# e-Mail address.
def trimName(address):
    return(address.replace('from_','').replace('.txt',''))

    
""" 
E-Mail Parser
"""
# The e-mail parser that itereates through the selected e-mails and creates a 
# set of pkl files with summary data. 
def parseMailFiles():
    # Create empty lists to store various result sets
    addresses = {}
    from_data = []
    word_data = []
    poi_data = []
    toParse = []
    
    # Get a list of all the available sender e-mail files, iterate through them
    # and build a dict of the senders we can work with later.
    for sender in glob.glob(mailListPath + "from*.txt"):
        mailAddress = trimName(sender.split('/')[-1])
        addresses[mailAddress] = {'FromPath': sender, 'POI': 0}
    # Next we look at how many of our address list are in our base data set and
    # how many are identified as persons of interest.
    pois = 0
    cnt = 0
    
    for person in data_dict:
        # Store the persons e-mail address and if not null, update the corresp-
        # onding Address record.
        eml = data_dict[person]['email_address']
        if eml != 'NaN':
            if eml in addresses:
                addresses[eml]['POI'] = data_dict[person]['poi']
                cnt += 1
                if data_dict[person]['poi']:
                    pois += 1
                # We want to ensure that everyone in the dataset is included
                # in the list of e-mails that we will parse so we ensure they
                # are added to the toParse list. 
                toParse.append(eml)
       
    # To try and keep the ratio of POIs to Non-POI users consistent with the 
    # main sample and to keep processing time reasonable, I am taking the POIs
    # and 10 times that in non-POI e-mails for parsing.
    for sender in addresses:
        if addresses[sender]['POI'] == 0 and cnt <= (10 * pois) and sender not in toParse:
            toParse.append(sender)
            cnt += 1
    # Some more counters
    temp_counter = 0
    total_mails = 0
    # Iterate through the e-mails of the selected senders
    for address in toParse:
        mlcount = 0
        fromMails = open(addresses[address]['FromPath'],'r')
        temp_counter += 1
        for name, from_person in [(address, fromMails)]:
            # Iterate through each person and each mail
            for path in from_person:
                path = path.replace('enron_mail_20110402/', 
                mailDir).replace('\n', '').replace('\r', '')
                try:                
                    # For each mail, we need to handle text cleanup and add it
                    # to the main word data                    
                    email = open(path, "r")
                    t = parseOutText(email)
                    total_mails += 1
                    if t[0] == ' ':
                        t = t[1:]
                    t = t.strip() 
                    t = t.replace('\n', ' ').replace('\r', '')
                    # Remove self identifying common words
                    t = t.replace(u'0516pm', '')
                    t = t.replace(u'cc', '')
                    t = t.replace(u'issu', '')
                    t = t.replace(u'forward', '')
                    t = t.replace(u'enron', '')
                    t = t.replace(u'chang', '')
                    t = t.replace(u'john', '')
                    t = t.replace(u'ani', '')
                    t = t.replace(u'david', '')
                    t = t.replace(u'nonprivilegedpst', '')
                    t = t.replace(u'messag', '')
                    t = t.replace(u'  ', ' ')
                    t = ' '.join(t.split())
                    # Check for uniqueness and add
                    if t not in word_data:
                        word_data.append(t)
                        poi_data.append(addresses[address]['POI'])
                        from_data.append(address)
                        mlcount += 1
                    email.close()
                except (IOError):
                    print('Bad path:',path)
        print('Process: '+address, 'Is POI:', addresses[address]['POI'], 
              mlcount, "{0:.2f}%".format((temp_counter/(10.0 * pois))*100))
    # Dump pickes so I don't have to re-parse the mails each time.
    pickle.dump( word_data, open("p5_parsed_word_data.pkl", "w") )
    pickle.dump( poi_data, open("p5_pois.pkl", "w") )
    pickle.dump( from_data, open("p5_email_authors.pkl", "w") )

""" TLIDF Vectorizer 
Based on the example from the text learning module, the example in the forum and
the How I Met Your Mother SKLearn text learning on the web.
"""

def prepClassifier():
    sys.path.append( "../tools/" )
    # Parsed from emails previously
    words_file = mypath + "p5_parsed_word_data.pkl" 
    poi_file = mypath + "p5_pois.pkl"
    words = pickle.load( open(words_file, "r"))
    poi_data = pickle.load( open(poi_file, "r") )
    # Create a test and train split    
    features_train, features_test, labels_train, labels_test = cv.train_test_split(
        words, poi_data, test_size=0.5, random_state=42)
    # text vectorization--go from strings to lists of numbers
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=0.20, max_df=0.75,
                                  stop_words='english')
    # Train and transform the training data and then transform the test set
    features_train = vectorizer.fit_transform(features_train)
    features_test  = vectorizer.transform(features_test)
    # Feature selection for the most important words
    selector = SelectPercentile(f_classif, percentile=50)
    selector.fit(features_train, labels_train)
    features_transformed = selector.transform(features_train)
    features_test_transformed  = selector.transform(features_test)
    # Now the decision tree for picking the POIs
    clf = DecisionTreeClassifier()
    clf = clf.fit(features_transformed, labels_train)
    acc = clf.score(features_test_transformed, labels_test)
    # Some accuracy reporting and print out the top 10 words
    print('Original accuracy:', acc)
    scores = clf.feature_importances_
    indices = np.argsort(scores)[::-1]
    wrds = vectorizer.get_feature_names()
    for f in range(len(indices)):
        print(scores[indices[f]], wrds[indices[f]])
    # Return the important data
    return(words, selector, vectorizer, clf)

# TestSenders
# Submits each sender's emails to be vectorized and the number of flagged/POI
# e-mails to be returned in a pkl object
def testSenders(words, selector, vectorizer, clf):
    auth_file =  mypath + "/p5_email_authors.pkl"
    auths = pickle.load(open(auth_file,'r'))
    results = {}    
    toTest = {}
    # Initialize our test set
    for cnt in range(0,len(auths)):
        if auths[cnt] not in toTest:
            toTest[auths[cnt]] = []
        toTest[auths[cnt]].append(words[cnt])
    # And now test each author
    for auth in toTest:
        vct = vectorizer.transform(toTest[auth])
        vct = selector.transform(vct)
        flagged = sum(clf.predict(vct))
        results[auth] = {'mails': vct.shape[0], 'flaggedMails': flagged}
    # Now save the resulting predictions
    pickle.dump( results, open("p5_analyzedEmailData.pkl", "w") )

# Since these take a long time to run, uncomment the one that is appropriate
#parseMailFiles()
words, sel, vect, clf = prepClassifier()        
testSenders(words, sel, vect, clf)
