#! /usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Name:         TEF_soln.py
# Author:       Yixia Mao <ym943@nyu.edu>
# Created:      Fri Nov 15 2013 by Yixia Mao (<ym943@nyu.edu>)
#----------------------------------------------------------------------------
from __future__ import with_statement, division
import sys, csv, re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
#from scipy.stats.distributions import entropy
from sklearn.utils import shuffle
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import math
from sklearn.cross_validation import ShuffleSplit


CLASS_NAMES = ("ham", "spam")
CLASSES = (0, 1)
CLASS_IDX = dict(zip(CLASS_NAMES, CLASSES))
POS_CLASS_NAME = "spam"
POS_CLASS = CLASS_IDX[POS_CLASS_NAME]

N_FOLDS = 10

# For feature selection, how many to choose
BEST_FEATURES = 4000

#  These delimiters are taken from the homework assigned, with a few necessary
#  backslashes added.
DELIMITERS = re.compile(r'[.,:;\'"?!@#$%^&*\n\t {}|[\]\<>/`~1234567890=_+()\\-]+')


textfile = "spam_data_Text.csv"

def cross_validate(X, y, binary):
    """Cross-validation engine.
        
        Inputs: X, a doc/word matrix.  y, an array of class labels.  binary, a
        flag indicating whether binomial (Bernoulli) NB or multinomial NB should
        be used.
        
        Performs k-fold cross-validation (directed by global variable N_FOLDS).
        Prints the accuracy and AUC of each fold, and finally the mean and stdev
        of the values.
        """
    accuracies = []
    AUCs = []
    confusions = []
    for (fold_i, (train, test)) in enumerate(KFold(len(y), n_folds=N_FOLDS)):
        print
        print "Fold: ", fold_i
        (X_train, X_test) = X[train], X[test]
        (y_train, y_test) = y[train], y[test]
        # Train and test the classifier
        if binary:
            clf = BernoulliNB()
        else:
            clf = MultinomialNB()
        clf.fit(X_train.toarray(), y_train)
        y_pred = clf.predict(X_test.toarray())
        acc = accuracy_score(y_test, y_pred)
        print "Accuracy:" ,acc
        accuracies.append(acc)
        # See:
        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
        # predict_proba returns a matrix of the probabilities for each class.
        # For roc_auc_score we have to pass in just the positive class probs.
        y_scores = clf.predict_proba(X_test)[:,POS_CLASS]
        AUC = roc_auc_score(y_test, y_scores)
        print "AUC:", AUC
        AUCs.append(AUC)
        confusions.append(confusion_matrix(y_test, y_pred))
    
    # CV is done.  Convert accuracies and AUCs to numpy arrays to get stats.
    accuracies = np.array(accuracies)
    AUCs = np.array(AUCs)
    
    total_confusion = np.zeros((2,2))
    for i in xrange(len(confusions)):
        total_confusion = np.add(total_confusion, confusions[i])
    #mean_confusion = mean_confusion/len(confusions)
    
    print "From", N_FOLDS, "folds:"
    print "\taccuracy =", accuracies.mean(), "+-", accuracies.std()
    print "\tAUC =", AUCs.mean(), "+-", AUCs.std()
    print "Confusion Matrix: Row i is ACTUALLY in class i, column j is PREDICTED to be in class j"
    print total_confusion

def outputPredictions(pos_probs, targ_test, filename):
    with open(filename, 'w') as test_file:
        file_writer = csv.writer(test_file)
        file_writer.writerow(['Actual', 'Probability'])
        for i in xrange(pos_probs.shape[0]):
            actual = targ_test[i]
            probability = pos_probs[i]
            file_writer.writerow([repr(actual)+','+repr(probability)])

def train_test_split(X,y,binary,test_percent):
    mySplit = ShuffleSplit(n = y.shape[0], n_iter=1, test_size = test_percent, indices=True)
    for train, test in mySplit:
        feat_train = X[train]
        feat_test = X[test]
        targ_train = y[train]
        targ_test = y[test]
        
        if binary:
            clf = BernoulliNB()
        else:
            clf = MultinomialNB()
        
        thisFit = clf.fit(feat_train,targ_train)
        myPredictions = thisFit.predict(feat_test)
        probs = thisFit.predict_proba(feat_test)
        pos_probs = probs[:,1]
        
        print 'Accuracy = '+repr(accuracy_score(targ_test, myPredictions))
        print 'ROC = '+repr(roc_auc_score(targ_test, pos_probs))
        print 'Confusion matrix: '+repr(confusion_matrix(targ_test, myPredictions))
    
    outputPredictions(pos_probs, targ_test, 'hw4Probs.csv')
                                     

def tokenize(text):
    return [tok.strip().lower() for tok in DELIMITERS.split(text)]

def infoGain(X,y,features):
    class1prob = np.sum(y)/y.shape
    wholeEntropy = -class1prob*math.log(class1prob,2)-(1-class1prob)*math.log(1-class1prob,2)
    yList = y.tolist()
    yInds = [i for i in xrange(len(yList)) if yList[i] == 1]
    negInds = [i for i in xrange(len(yList)) if yList[i] == 0]
    
    num1 = np.sum(y)
    num0 = y.shape - num1
    
    IGs = []
    
    for i in xrange(X.shape[1]):
        mySlice = np.array(X[:,i].todense())
        posProb = np.sum(mySlice)/mySlice.shape[0]
        negProb = 1-posProb
        
        numPos = np.sum(mySlice)
        numNeg = mySlice.shape[0] - numPos
        
        Class1 = mySlice[yInds]
        
        class1pos = float(np.sum(Class1))
        class1neg = float(np.sum(y) - class1pos)
        
        Class0 = mySlice[negInds]
        
        class0pos = float(np.sum(Class0))
        class0neg = (y.shape - np.sum(y)) - class0pos
        class0neg = float(class0neg[0])
        
        
        c1p = class1pos*1.0/(class1pos+class0pos)
        c0p = class0pos*1.0/(class1pos+class0pos)
        
        c1n = class1neg*1.0/(class1neg+class0neg)
        c0n = class0neg*1.0/(class1neg+class0neg)
        
        if c1p>0 and c0p>0:
            posEnt = -c1p*math.log(c1p,2)-c0p*math.log(c0p,2)
        else:
            posEnt = 0
        if c1n > 0 and c0n>0:
            negEnt = -c1n*math.log(c1n,2)-c0n*math.log(c0n,2)
        else:
            negEnt = 0
        
        myEntropy = (wholeEntropy - (posProb*posEnt+negProb*negEnt))[0]
        
        #print 'feature name is '+repr(features[i])+' IG is '+repr(myEntropy)
        
        IGs.append(myEntropy)
    
    return IGs

def topFeatures(X, infoGain, features, printing=True):
    infoGain = -1*np.array(infoGain)
    myInds = np.argsort(infoGain)
    myInds = myInds[0:BEST_FEATURES]
    
    feat_array = np.array(features)
    IG_array = np.array(infoGain)
    
    X_reduced = X[:, myInds]
    feats_reduced = feat_array[myInds.tolist()]
    IGs_reduced = IG_array[myInds.tolist()]
    
    for i in xrange(feats_reduced.shape[0]):
        print 'Feature name is '+feats_reduced[i].decode('ascii', 'ignore')+' , info gain is '+repr(-1*IGs_reduced[i])
    
    return X_reduced

def text_feature_engineering(filename, binary):
    
    """Given a CSV file consisting of two instances: text,classname reads it in and tokenizes the whole thing.  Keyword binary determines whether tokenizer uses binary values or word frequency counts in the returned matrix X.
        
        Returns three values: the tokenizer (so words and stats can be
        extracted from it), the X doc/word sparse matrix, and an array of classes. """
    docs=[]
    targets=[]
   
    with open(filename, 'rb') as infile:
        counter = 0
        for (text,classname) in csv.reader(infile,doublequote=False,quotechar="'",escapechar="\\"):
            if classname != '@class@':
                docs.append(text.decode('ascii', 'ignore'))
                targets.append(CLASS_IDX[classname])

    vectorizer = CountVectorizer(tokenizer=tokenize,stop_words='english', binary=binary, min_df=5)
    X = vectorizer.fit_transform(docs)
    features = vectorizer.get_feature_names()
    y = np.array(targets)
    
    return (X, y, features)


def main():
    
    # First create a binary vectorizer from the textfile
    print "===== BINARY representation goes here"

    (X_binary, y_binary, feats_binary) = text_feature_engineering(textfile, binary=False)
    (X_binary, y_binary) = shuffle(X_binary, y_binary)
    # cross_validate(X_binary, y_binary, binary=True)
    
    # Now create a multinomial vectorizer and do it over
    print "\n===== MULTINOMIAL representation"
    
    # cross_validate(X_binary, y_binary, binary=False)
    # binary = False
    # test_percent = 0.34
    # train_test_split(X_binary,y_binary,binary,test_percent)
    
    # Now do feature selection
    print "\n===== Selecting top %d features" % BEST_FEATURES
    
    IGs = infoGain(X_binary, y_binary, feats_binary)
    X_reduced = topFeatures(X_binary,IGs,feats_binary,printing=True)
    cross_validate(X_reduced, y_binary, binary=False)
    
    #Now compare training on past, testing on future to training/testing randomly
    print "\n===== Comparing train on past, test on future to random split"




main()
