# Choong-Wan Woo
from numpy import array, zeros, where, sum

kINSP = array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = array([(-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])


def weight_vector(x, y, alpha):
    """
    Given a vector of alphas, compute the primal weight vector.
    """
    # TODO: IMPLEMENT THIS FUNCTION
    w = zeros(len(x[0]))
    w = sum([alpha[ii]*x[ii]*y[ii] for ii in xrange(len(x))], axis=0)
    return w


def find_support(x, y, w, b, tolerance=0.001):
    """
    Given a primal support vector, return the indices for all of the support
    vectors
    """
    # TODO: IMPLEMENT THIS FUNCTION
    aa=w*x
    bb=aa.sum(axis=1)+b
    cc=where(y*(bb-y)<tolerance)
    support = set([ii for ii in cc[0]])
    return support


def find_slack(x, y, w, b):
    """
    Given a primal support vector instance, return the indices for all of the
    slack vectors
    """
    aa=w*x
    bb=aa.sum(axis=1)+b
    cc = where(y*(bb-y)<0)
    slack = set([ii for ii in cc[0]])
    # TODO: IMPLEMENT THIS FUNCTION
    return slack
