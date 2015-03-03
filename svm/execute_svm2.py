import os

# 1. Use the scikit implementation of support vector machines to train a classifier to distinguish 3's from 8's. (Use the MNIST data from the KNN homework.) --- Okay done. see https://github.com/wanirepo/ml-hw/blob/master/svm/svm2.py

# 2. Try at least five values of the regularization parameter C and at least two kernels.

# os.system("python svm2.py --C=1 --kernel='linear' --limit=400")
# os.system("python svm2.py --C=2 --kernel='linear' --limit=400")
# os.system("python svm2.py --C=4 --kernel='linear' --limit=400")
# os.system("python svm2.py --C=8 --kernel='linear' --limit=400")
# os.system("python svm2.py --C=16 --kernel='linear' --limit=400")

os.system("python svm2.py --C=1 --kernel='poly' --gamma=1 --degree=1 --limit=400")
# os.system("python svm2.py --C=2 --kernel='rbf' --gamma=1 --limit=400 ")
# os.system("python svm2.py --C=4 --kernel='rbf' --gamma=1 --limit=400 ")
# os.system("python svm2.py --C=8 --kernel='rbf' --gamma=1 --limit=400 ")
# os.system("python svm2.py --C=16 --kernel='rbf' --gamma=1 --limit=400 ")

# Give examples of support vectors with a linear kernel.

#clf.support_vectors_
