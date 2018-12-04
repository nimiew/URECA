"""RBF model"""
import random
import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X = np.load("X.npy") #(823, 34716)
X = np.nan_to_num(X)
Y = np.load("Y.npy") #(823,)

result = [] #Store the accuracy for each run so we can take the average later
epoch = 30

#Find average accuracy of 30 runs using RBF
for j in range(30):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)
        svclassifier = SVC(kernel='rbf', gamma='auto')
        svclassifier.fit(X_train, Y_train)
        Y_pred = svclassifier.predict(X_test)
        print(accuracy_score(Y_test, Y_pred))
        result.append(accuracy_score(Y_test, Y_pred))
acc = sum(result)/float(len(result))
print("rbf acc %.10g"%(acc))
