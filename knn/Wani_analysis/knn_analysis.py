import argparse
from collections import Counter, defaultdict

import random
import numpy
from numpy import median
from sklearn.neighbors import BallTree

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
        f.close()


class Knearest:
    """
    kNN classifier
    """

    def __init__(self, x, y, k=5):
        """
        Creates a kNN instance

        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        # You can modify the constructor, but you shouldn't need to
        self._kdtree = BallTree(x)
        self._y = y
        self._k = k

    def majority(self, item_indices):
        """
        Given the indices of training examples, return the majority label.  If
        there's a tie, return the median value (as implemented in numpy).

        :param item_indices: The indices of the k nearest neighbors
        """

	# Finish this function to return the most common y value for
	# these indices
	
	b = Counter(self._y[item_indices])
	vals = numpy.array(b.values())
	keys = numpy.array(b.keys())
	whmax = [i for i,j in enumerate(vals) if j == max(vals)]
		
	return median(keys[whmax])

    def classify(self, example):
        """
        Given an example, classify the example.

        :param example: A representation of an example in the same
        format as training data
        """
	# Finish this function to find the k closest points, query the
	# majority function, and return the value.
	
	dist, indices = self._kdtree.query(example, k=self._k) 
	return self.majority(indices[0])

    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrixfor the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.

        :param test_x: Test data representation
        :param test_y: Test data answers
        """

        # Finish this function to build a dictionary with the
        # mislabeled examples.  You'll need to call the classify
        # function for each example.

        d = defaultdict(dict)
	for x in xrange(10):
	    for y in xrange(10):
		d[x][y] = 0

        data_index = 0
        for xx, yy in zip(test_x, test_y):
            d[yy][int(round(self.classify(xx)))] += 1
	    
#-            if data_index % 100 == 0:
#-	        print("%i/%i for confusion matrix" % (data_index, len(test_x)))
	    data_index += 1
        return d

    @staticmethod
    def acccuracy(confusion_matrix):
        """
        Given a confusion matrix, compute the accuracy of the underlying classifier.
        """

        # You do not need to modify this function

        total = 0
        correct = 0
        for ii in confusion_matrix:
            total += sum(confusion_matrix[ii].values())
            correct += confusion_matrix[ii].get(ii, 0)

        if total:
            return float(correct) / float(total)
        else:
            return 0.0


if __name__ == "__main__":
    k_choices = [1,3,5,9,15]
    limit_choices = [50,100,200,300,500]
    for kk in k_choices:
	for jj in limit_choices:
	    parser = argparse.ArgumentParser(description='KNN classifier options')
	    parser.add_argument('--k', type=int, default=kk,
	                        help="Number of nearest points to use")
	    parser.add_argument('--limit', type=int, default=jj,
	                        help="Restrict training to this many examples")
	    args = parser.parse_args()

	    data = Numbers("../data/mnist.pkl.gz")

	    # You should not have to modify any of this code

	    if args.limit > 0:
	        print("Data limit: %i" % args.limit)
	        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
	                       args.k)
	    else:
	        knn = Knearest(data.train_x, data.train_y, args.k)
	    print("Done loading data")

	    confusion = knn.confusion_matrix(data.test_x, data.test_y)
	    print("\t" + "\t".join(str(x) for x in xrange(10)))
	    print("".join(["-"] * 90))
	    for ii in xrange(10):
	        print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
	                                       for x in xrange(10)))
	    print("Accuracy: %f" % knn.acccuracy(confusion))

	    s = "Limit = " + repr(args.limit) + ", k = " + repr(args.k)
	    print s
	    for x in xrange(10):
		confusion[x][x] = 0
	    maxa = 0
	    for x in xrange(10):
		a = numpy.array(confusion[x].items())
		maxa = max(max(a[:,1]), maxa)
		if maxa == max(a[:,1]):
		    ind = [x, a[:,1].argmax()]
	    s = "Most confusing pairs are " + repr(ind[0]) + " and " + repr(ind[1]) + ": " + repr(maxa) + "."
	    print s




