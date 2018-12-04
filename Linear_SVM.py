"""Linear SVM model"""
import random
import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X = np.load("X.npy") #(823, 34716)
X = np.nan_to_num(X)
Y = np.load("Y.npy") #(823,)

result = []
epoch = 30
#Find average accuracy of 30 runs using linear svm
for j in range(30):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)
        svclassifier = SVC(kernel='linear', gamma='auto')
        #print('Fitting')
        svclassifier.fit(X_train, Y_train)
        Y_pred = svclassifier.predict(X_test)
	print(accuracy_score(Y_test, Y_pred))  
        result.append(accuracy_score(Y_test, Y_pred))
acc = sum(result)/float(len(result))
print("Linear acc %.10g"%(acc))
