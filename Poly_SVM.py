"""Polynomial SVM model"""
import random
import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X = np.load("X.npy") #(823, 34716)
X = np.nan_to_num(X)
Y = np.load("Y.npy") #(823,)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)
result = [] #Store accuracy for each run
acc = [] #Store accuracy for each degree

#Find average accuracy of 10 runs for each degree from 1-10  
for i in range(10):
    result.append([]) #array to hold accuracies for degree i+1
    for j in range(10): #10 runs
        svclassifier = SVC(kernel='poly', degree=i+1, gamma='auto')
        print('Fitting')
        svclassifier.fit(X_train, Y_train)
        Y_pred = svclassifier.predict(X_test)  
        result[i].append(accuracy_score(Y_test, Y_pred))
    acc.append(sum(result[i]/float(len(result[i])))) #Store accuracy for degree i+1
for i in range(10):
    print("Degree %d acc %f"%(i+1, acc[i]))
