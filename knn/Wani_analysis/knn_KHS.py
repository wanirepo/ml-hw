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
        assert len(item_indices) == self._k, "Did not get k inputs"
        temp = []       
        for ii in item_indices:
			temp.append(self._y[ii])
       
		# Find frequency for each y : z = [[y, frequency],,]
        x = numpy.array(temp) 
        unique = numpy.unique(x)
        i = 0; j = 0; count = 0; z = []
        while i < len(unique):
			while j < len(temp):
				if unique[i] == temp[j]:
					count +=1
				j += 1
			z.append([unique[i], count])
			count = 0
			j = 0; i += 1
				    		
        # Find the most frequent y 
        i = 0
        max_freq = z[0][1]; max_y = z[0][0]
        while i < len(z): 
			if z[i][1] > max_freq:
				max_freq = z[i][1]; max_y = z[i][0]
			i += 1
		
		# Check if there's a tie
		# Finish this function to return the most common y value for these indices
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html
        i = 0; j = 0
        while i < len(z): 
			if z[i][1] == max_freq:
				j += 1
			i += 1
        if j == 1:
			return max_y
        else:
			return numpy.median(x)

    def classify(self, example):
        """
        Given an example, classify the example.
        :param example: A representation of an example in the same format as training data
        """
        dist, ind = self._kdtree.query(example, k = self._k)
        return self.majority(list(ind[0]))
        # Finish this function to find the k closest points, query the
        # majority function, and return the value.


    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrix for the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.
        :param test_x: Test data representation
        :param test_y: Test data answers
        """
        # Finish this function to build a dictionary with the
        # mislabeled examples.  You'll need to call the classify
        # function for each example.
        d = defaultdict(dict)
        data_index = 0       
        i, j = 0, 0
        while i < 10:
			while j < 10:
				d[i][j] = 0; j += 1
			j = 0; i += 1   
			    
        for xx, yy in zip(test_x, test_y):
            r = self.classify(xx)
            d[r][yy] = d[r][yy] + 1			
            data_index += 1            
            if data_index % 100 == 0:
                print("%i/%i for confusion matrix" % (data_index, len(test_x)))
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
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()
    data = Numbers("../data/mnist.pkl.gz")

    # You should not have to modify any of this code
    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit], args.k)
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
