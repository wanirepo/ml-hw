import os

#  Try at least five values of the regularization parameter C and at least two kernels.
os.system("python svm2.py --C=1 --kernel='linear'")
os.system("python svm2.py --C=2 --kernel='linear'")
os.system("python svm2.py --C=4 --kernel='linear'")
os.system("python svm2.py --C=8 --kernel='linear'")
os.system("python svm2.py --C=16 --kernel='linear'")

os.system("python svm2.py --C=1 --kernel='rbf' --gamma=10")
os.system("python svm2.py --C=2 --kernel='rbf' --gamma=10")
os.system("python svm2.py --C=4 --kernel='rbf' --gamma=10")
os.system("python svm2.py --C=8 --kernel='rbf' --gamma=10")
os.system("python svm2.py --C=16 --kernel='rbf' --gamma=10")
