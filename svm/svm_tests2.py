import argparse
from collections import Counter, defaultdict

import random
import numpy
from sklearn import svm

import cPickle, gzip

f = gzip.open("/Users/clinpsywoo/github/ml-hw/data/mnist.pkl.gz", 'rb')
train_set, valid_set, test_set = cPickle.load(f)

train_x, train_y = train_set
test_x, test_y = valid_set

# get 3 and 8
idx_train = numpy.where((train_y==3) | (train_y==8))
train_x = train_x[idx_train]
train_y = train_y[idx_train]

idx_test = numpy.where((test_y==3) | (test_y==8))
test_x = test_x[idx_test]
test_y = test_y[idx_test]

clf = svm.SVC(C=1, kernel='linear', degree=0, coef0=0, gamma=0)
clf.fit(train_x, train_y)
predicted_y=clf.predict(test_x)
acc = float(sum(predicted_y==test_y))/float(len(test_y))*100
print("accuracy = %.2f%%" %acc)