#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def gen_train_test_data():
    test_ratio = .3 
    df = pandas.read_csv('./yelp_labelled.txt', sep='\t', header=None, encoding='utf-8')
    count_vect = CountVectorizer()
    X = count_vect.fit_transform(df[0])
    y = df[1].tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=0)
    return X_train, X_test, y_train, y_test


def multinomial_nb(X_train, X_test, y_train, y_test):
    # TODO: fill this function
    # train by X_train and y_train
    # report the predicting accuracy for both the training and the test data 
    p_pos = y_train.count(1) / len(y_train) # prior probability of text class = 1
    p_neg = y_train.count(0) / len(y_train) # prior probability of text class = 0

    pos_idx = [i for i in range(len(y_train)) if y_train[i] == 1]
    neg_idx = [i for i in range(len(y_train)) if y_train[i] == 0]
    num_words_pos = np.sum(X_train[pos_idx])
    num_words_neg = np.sum(X_train[neg_idx])
    pos_cond_prob = (1 + np.sum(X_train[pos_idx],axis=0)) / (num_words_pos + X_train.shape[1]) # calculate conditional probability of each words 
    neg_cond_prob = (1 + np.sum(X_train[neg_idx],axis=0)) / (num_words_neg + X_train.shape[1])

    def inference(sentence_vec,pos_cond_prob,neg_cond_prob,y_true):
        pos_prob_init = np.multiply(pos_cond_prob,sentence_vec.toarray())
        neg_prob_init = np.multiply(neg_cond_prob,sentence_vec.toarray())
        pos_prob = [word[word>0] for word in pos_prob_init]
        pos_prob = np.array([p_pos*np.prod(row) for row in pos_prob]).transpose()
        neg_prob = [word[word>0] for word in neg_prob_init]
        neg_prob = np.array([p_neg*np.prod(row) for row in neg_prob]).transpose()
        y_pred = np.where(pos_prob - neg_prob > 0, 1, 0)
        acc = accuracy_score(y_true,y_pred)
        roc_auc = roc_auc_score(y_true,y_pred)
        f1 = f1_score(y_true,y_pred)
        return {'accuracy':acc,'roc_auc':roc_auc,'f1_score':f1}
        
    y_train_performance = inference(X_train,pos_cond_prob,neg_cond_prob,y_train)
    print('Training preformance of Multinomial NB:',y_train_performance)
    y_test_performance = inference(X_test,pos_cond_prob,neg_cond_prob,y_test)
    print('Testing performance of Multinomial NB',y_test_performance)


def bernoulli_nb(X_train, X_test, y_train, y_test):
    # TODO: fill this function
    # train by X_train and y_train
    # report the predicting accuracy for both the training and the test data
    
    X_train = (X_train.toarray() > 0).astype(int)
    pos_count = np.count_nonzero(y_train)
    neg_count = len(y_train) - pos_count

    pos_log_prior = np.log(pos_count / len(y_train))
    neg_log_prior = np.log(neg_count / len(y_train))
    pos_idx = [i for i in range(len(y_train)) if y_train[i] == 1]
    neg_idx = [i for i in range(len(y_train)) if y_train[i] == 0]
    pos_cond_prob = (1 + np.sum(X_train[pos_idx],axis=0)) / (pos_count + 2) # calculate conditional probability of each words 
    neg_cond_prob = (1 + np.sum(X_train[neg_idx],axis=0)) / (neg_count + 2)

    def inference(sentence_vec,pos_cond_prob,neg_cond_prob,y_true):
        pos_log_likelihood = pos_log_prior + np.dot(sentence_vec,np.log(pos_cond_prob)) + np.log(1-pos_cond_prob).sum()
        neg_log_likelihood = neg_log_prior + np.dot(sentence_vec,np.log(neg_cond_prob)) + np.log(1-neg_cond_prob).sum()
        y_pred = np.where(pos_log_likelihood-neg_log_likelihood > 0,1,0)
        acc = accuracy_score(y_true,y_pred)
        roc_auc = roc_auc_score(y_true,y_pred)
        f1 = f1_score(y_true,y_pred)
        return {'accuracy':acc,'roc_auc':roc_auc,'f1_score':f1}

    y_train_performance = inference(X_train,pos_cond_prob,neg_cond_prob,y_train)
    print('Training preformance of Bernoulli NB:',y_train_performance)
    y_test_performance = inference(X_test.toarray(),pos_cond_prob,neg_cond_prob,y_test)
    print('Testing performance of Bernoulli NB:',y_test_performance)    


def main(argv):
    X_train, X_test, y_train, y_test = gen_train_test_data()

    multinomial_nb(X_train, X_test, y_train, y_test)
    bernoulli_nb(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main(sys.argv)


