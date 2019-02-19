import random
import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

np.random.seed(10)

print('823 samples')
X = np.load("X_old.npy")
X = np.nan_to_num(X)
Y = np.load("Y_old.npy")
accuracy_list = []
sensitivity_list = []
specificity_list = []
f1_score_list = []
for i in range(10):
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, stratify=Y)
	svclassifier = SVC(kernel='linear', gamma='auto')
	#print('Fitting epoch', i)
	svclassifier.fit(X_train, Y_train)
	Y_pred = svclassifier.predict(X_test)
	print(type(Y_pred[0]))
	print(Y_pred.shape)
	exit()
	accuracy_list.append(accuracy_score(Y_test, Y_pred))
	c_matrix = confusion_matrix(y_true=Y_test,y_pred=Y_pred)
	TN = float(c_matrix[0][0])
	FP = float(c_matrix[0][1])
	FN = float(c_matrix[1][0])
	TP = float(c_matrix[1][1])
	sensitivity = TP/(TP+FN)
	specificity = TN/(TN+FP)
	f1_score = 2*TP/(2*TP+FP+FN)
	sensitivity_list.append(sensitivity)
	specificity_list.append(specificity)
	f1_score_list.append(f1_score)
print('linear')
print('sanity check, len is: ', len(accuracy_list))
print('accuracy: ', sum(accuracy_list)/float(len(accuracy_list)))
print('sensitivity: ', sum(sensitivity_list)/float(len(sensitivity_list)))
print('specificity: ', sum(specificity_list)/float(len(specificity_list)))
print('f1_score: ', sum(f1_score_list)/float(len(f1_score_list)))

accuracy_list = []
sensitivity_list = []
specificity_list = []
f1_score_list = []
for i in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, stratify=Y)
        svclassifier = SVC(kernel='poly', gamma='auto')
        #print('Fitting epoch', i)
        svclassifier.fit(X_train, Y_train)
        Y_pred = svclassifier.predict(X_test)
        accuracy_list.append(accuracy_score(Y_test, Y_pred))
        c_matrix = confusion_matrix(y_true=Y_test,y_pred=Y_pred)
        TN = float(c_matrix[0][0])
        FP = float(c_matrix[0][1])
        FN = float(c_matrix[1][0])
        TP = float(c_matrix[1][1])
        sensitivity = TP/(TP+FN)
        specificity = TN/(TN+FP)
        f1_score = 2*TP/(2*TP+FP+FN)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        f1_score_list.append(f1_score)
print('poly')
print('sanity check, len is: ', len(accuracy_list))
print('accuracy: ', sum(accuracy_list)/float(len(accuracy_list)))
print('sensitivity: ', sum(sensitivity_list)/float(len(sensitivity_list)))
print('specificity: ', sum(specificity_list)/float(len(specificity_list)))
print('f1_score: ', sum(f1_score_list)/float(len(f1_score_list)))

accuracy_list = []
sensitivity_list = []
specificity_list = []
f1_score_list = []
for i in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, stratify=Y)
        svclassifier = SVC(kernel='rbf', gamma='auto')
        #print('Fitting epoch', i)
        svclassifier.fit(X_train, Y_train)
        Y_pred = svclassifier.predict(X_test)
        accuracy_list.append(accuracy_score(Y_test, Y_pred))
        c_matrix = confusion_matrix(y_true=Y_test,y_pred=Y_pred)
        TN = float(c_matrix[0][0])
        FP = float(c_matrix[0][1])
        FN = float(c_matrix[1][0])
        TP = float(c_matrix[1][1])
        sensitivity = TP/(TP+FN)
        specificity = TN/(TN+FP)
        f1_score = 2*TP/(2*TP+FP+FN)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        f1_score_list.append(f1_score)
print('rbf')
print('sanity check, len is: ', len(accuracy_list))
print('accuracy: ', sum(accuracy_list)/float(len(accuracy_list)))
print('sensitivity: ', sum(sensitivity_list)/float(len(sensitivity_list)))
print('specificity: ', sum(specificity_list)/float(len(specificity_list)))
print('f1_score: ', sum(f1_score_list)/float(len(f1_score_list)))

print('1035 samples')
X = np.load("X.npy")
X = np.nan_to_num(X)
Y = np.load("Y.npy")
accuracy_list = []
sensitivity_list = []
specificity_list = []
f1_score_list = []
for i in range(10):
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, stratify=Y)
	svclassifier = SVC(kernel='linear', gamma='auto')
	#print('Fitting epoch', i)
	svclassifier.fit(X_train, Y_train)
	Y_pred = svclassifier.predict(X_test)
	accuracy_list.append(accuracy_score(Y_test, Y_pred))
	c_matrix = confusion_matrix(y_true=Y_test,y_pred=Y_pred)
	TN = float(c_matrix[0][0])
	FP = float(c_matrix[0][1])
	FN = float(c_matrix[1][0])
	TP = float(c_matrix[1][1])
	sensitivity = TP/(TP+FN)
	specificity = TN/(TN+FP)
	f1_score = 2*TP/(2*TP+FP+FN)
	sensitivity_list.append(sensitivity)
	specificity_list.append(specificity)
	f1_score_list.append(f1_score)
print('linear')
print('sanity check, len is: ', len(accuracy_list))
print('accuracy: ', sum(accuracy_list)/float(len(accuracy_list)))
print('sensitivity: ', sum(sensitivity_list)/float(len(sensitivity_list)))
print('specificity: ', sum(specificity_list)/float(len(specificity_list)))
print('f1_score: ', sum(f1_score_list)/float(len(f1_score_list)))

accuracy_list = []
sensitivity_list = []
specificity_list = []
f1_score_list = []
for i in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, stratify=Y)
        svclassifier = SVC(kernel='poly', gamma='auto')
        #print('Fitting epoch', i)
        svclassifier.fit(X_train, Y_train)
        Y_pred = svclassifier.predict(X_test)
        accuracy_list.append(accuracy_score(Y_test, Y_pred))
        c_matrix = confusion_matrix(y_true=Y_test,y_pred=Y_pred)
        TN = float(c_matrix[0][0])
        FP = float(c_matrix[0][1])
        FN = float(c_matrix[1][0])
        TP = float(c_matrix[1][1])
        sensitivity = TP/(TP+FN)
        specificity = TN/(TN+FP)
        f1_score = 2*TP/(2*TP+FP+FN)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        f1_score_list.append(f1_score)
print('poly')
print('sanity check, len is: ', len(accuracy_list))
print('accuracy: ', sum(accuracy_list)/float(len(accuracy_list)))
print('sensitivity: ', sum(sensitivity_list)/float(len(sensitivity_list)))
print('specificity: ', sum(specificity_list)/float(len(specificity_list)))
print('f1_score: ', sum(f1_score_list)/float(len(f1_score_list)))

accuracy_list = []
sensitivity_list = []
specificity_list = []
f1_score_list = []
for i in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, stratify=Y)
        svclassifier = SVC(kernel='rbf', gamma='auto')
        #print('Fitting epoch', i)
        svclassifier.fit(X_train, Y_train)
        Y_pred = svclassifier.predict(X_test)
        accuracy_list.append(accuracy_score(Y_test, Y_pred))
        c_matrix = confusion_matrix(y_true=Y_test,y_pred=Y_pred)
        TN = float(c_matrix[0][0])
        FP = float(c_matrix[0][1])
        FN = float(c_matrix[1][0])
        TP = float(c_matrix[1][1])
        sensitivity = TP/(TP+FN)
        specificity = TN/(TN+FP)
        f1_score = 2*TP/(2*TP+FP+FN)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        f1_score_list.append(f1_score)
print('rbf')
print('sanity check, len is: ', len(accuracy_list))
print('accuracy: ', sum(accuracy_list)/float(len(accuracy_list)))
print('sensitivity: ', sum(sensitivity_list)/float(len(sensitivity_list)))
print('specificity: ', sum(specificity_list)/float(len(specificity_list)))
print('f1_score: ', sum(f1_score_list)/float(len(f1_score_list)))
