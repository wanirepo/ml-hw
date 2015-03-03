import argparse
from collections import Counter, defaultdict

import random
import numpy as np
from sklearn import svm

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set

        idx_train=np.where((self.train_y==3)|(self.train_y==8))
        self.train_x = self.train_x[idx_train]
        self.train_y = self.train_y[idx_train]

	idx_test=np.where((self.test_y==3)|(self.test_y==8))
        self.test_x = self.test_x[idx_test]
        self.test_y = self.test_y[idx_test]
        f.close()

if __name__ == "__main__":
    #Try at least five values of the regularization parameter C and at least two kernels.
    #Give examples of support vectors with a linear kernel.
    parser = argparse.ArgumentParser(description='SVM classifier options')
    parser.add_argument('--C', type=float, default=1.0,
                        help="Penalty parameter C of the error term.")
    parser.add_argument('--kernel', type=str, default='linear',
                        help="Kernel methods, e.g., linear(default), poly, rbf, sigmoid, precomputed")
    parser.add_argument('--degree', type=int, default=1,
                        help="Degree of the polynomial kernel function (poly). Ignored by all other kernels.")
    parser.add_argument('--gamma', type=float, default=0.0,
                        help="Kernel coefficient for rbf, poly and sigmoid.")
    parser.add_argument('--coef0', type=float, default=0.0,
                        help="Independent term in kernel function. It is only significant in poly and sigmoid")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")    
    print("Done loading data")

    clf = svm.SVC(C=args.C, kernel=args.kernel, degree=args.degree, coef0=args.coef0, gamma=args.gamma)
    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        clf.fit(data.train_x[:args.limit], data.train_y[:args.limit])
    else:
	clf.fit(data.train_x, data.train_y)

    predicted_y = clf.predict(data.test_x)
    acc = float(sum(predicted_y==data.test_y))/float(len(data.test_y))*100
    print("**SETTINGS **********************")
    print "C: "+ repr(args.C) + ", kernel: " + args.kernel + ", degree: " + repr(args.degree)
    print "gamma: "+ repr(args.gamma) + ", coef0: " + repr(args.coef0)
    print("**Accuracy = %.2f%% ***************" %acc)



