'''
HW2 problem
'''

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.special as sp
import time
from scipy.optimize import minimize

import data_generator as dg

# you can define/use whatever functions to implememt

def get_scores(Wb, x, num_class, n, feat_dim):
    Wb_t = np.reshape(Wb, (-1, 1))
    W = np.reshape(Wb_t[:num_class * feat_dim], (num_class, feat_dim))
    b = np.reshape(Wb_t[-num_class:], (num_class, -1))
    return W @ np.reshape(x.T, (-1, n)) + b


########################################
# Part 1. cross entropy loss
########################################
def cross_entropy_softmax_loss(Wb, x, y, num_class, n, feat_dim):
    exp_scores = np.exp(get_scores(Wb, x, num_class, n, feat_dim))
    normalized_scores = exp_scores / exp_scores.sum(axis=0)

    real_scores = normalized_scores[y, np.arange(n)]
    return -1 * np.log(real_scores).sum()


########################################
# Part 2. SVM loss calculation
########################################
def svm_loss(Wb, x, y, num_class, n, feat_dim):
    scores = get_scores(Wb, x, num_class, n, feat_dim)
    real_scores = scores[y, np.arange(n)]

    losses = np.maximum(0, scores - real_scores + 1)
    return losses.sum() / n - 1

########################################
# Part 3. kNN classification
########################################
def knn_test(X_train, y_train, X_test, y_test, k):
    x_train_norms = np.sum(X_train ** 2, axis=1)[:, np.newaxis]
    x_test_norms = np.sum(X_test ** 2, axis=1)[:, np.newaxis].T
    train_test_dots = X_train @ X_test.T

    scores = x_train_norms + x_test_norms - 2 * train_test_dots
    sorted_scores = np.argsort(scores, axis=0)[:k, :]
    [ test_result ], _ = stats.mode(y_train[sorted_scores])

    return np.sum(test_result == y_test) / X_test.shape[0]


# now lets test the model for linear models, that is, SVM and softmax
def linear_classifier_test(Wb, x, y, num_class):
    n_test = x.shape[0]
    feat_dim = x.shape[1]
    
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:].squeeze()
    W = np.reshape(Wb[:-num_class], (num_class, feat_dim))
    accuracy = 0

    # W has shape (num_class, feat_dim), b has shape (num_class,)

    # score
    s = x@W.T + b
    # score has shape (n_test, num_class)
    
    # get argmax over class dim
    res = np.argmax(s, axis = 1)

    # get accuracy
    accuracy = (res == y).astype('uint8').sum()/n_test
    
    return accuracy


# number of classes: this can be either 3 or 4
num_class = 4

# sigma controls the degree of data scattering. Larger sigma gives larger scatter
# default is 1.0. Accuracy becomes lower with larger sigma
sigma = 1.0

print('number of classes: ',num_class,' sigma for data scatter:',sigma)
if num_class == 4:
    n_train = 400
    n_test = 100
    feat_dim = 2
else:  # then 3
    n_train = 300
    n_test = 60
    feat_dim = 2

# generate train dataset
print('generating training data')
x_train, y_train = dg.generate(number=n_train, seed=None, plot=True, num_class=num_class, sigma=sigma)

# generate test dataset
print('generating test data')
x_test, y_test = dg.generate(number=n_test, seed=None, plot=False, num_class=num_class, sigma=sigma)

# set classifiers to 'svm' to test SVM classifier
# set classifiers to 'softmax' to test softmax classifier
# set classifiers to 'knn' to test kNN classifier
classifiers = 'softmax'

if classifiers == 'svm':
    print('training SVM classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(svm_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))
    print('testing SVM classifier...')

    Wb = result.x
    print('accuracy of SVM loss: ', linear_classifier_test(Wb, x_test, y_test, num_class)*100,'%')

elif classifiers == 'softmax':
    print('training softmax classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(cross_entropy_softmax_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))

    print('testing softmax classifier...')

    Wb = result.x
    print('accuracy of softmax loss: ', linear_classifier_test(Wb, x_test, y_test, num_class)*100,'%')

else:  # knn
    # k value for kNN classifier. k can be either 1 or 3.
    k = 3
    print('testing kNN classifier...')
    print('accuracy of kNN loss: ', knn_test(x_train, y_train, x_test, y_test, k)*100
          , '% for k value of ', k)
