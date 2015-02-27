import argparse
from collections import Counter, defaultdict

import random
import numpy
from numpy import median
from sklearn.neighbors import BallTree

import cPickle, gzip

f = gzip.open("/Users/clinpsywoo/github/ml-hw/data/mnist.pkl.gz", 'rb')
train_set, valid_set, test_set = cPickle.load(f)

train_x, train_y = train_set
test_x, test_y = valid_set

# args.limit
train_x = train_x[:500]
train_y = train_y[:500]
k = 5

bt = BallTree(train_x)
dist, indices = bt.query(train_x, k=(k+1))
indices = numpy.delete(indices,0,1)
ylist = train_y[ind]

for i in range(0,indices.shape[0]):
	a = numpy.bincount(numpy.bincount(indices[i])) # If there a tie?
	if a[-1] > 1:
	else:
		b = numpy.bincount(ylist[i])
		whmax = b.argmax()
	
	
	counts = Counter(ylist[i]) # e.g., [3,1,0,0,3]
	a = counts.viewvalues()
	
	counts = numpy.bincount(numpy.bincount(ylist[i]))
	
	
############################################

bt = BallTree(train_x)
xx = test_x
yy = test_y

example = xx[0]
dist, item_indices = bt.query(example, k=5) 
