import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression

X = np.load("X_old.npy")
X = np.nan_to_num(X)
Y = np.load("Y_old.npy")

seed = 10
tf.set_random_seed(seed)
np.random.seed(seed)

trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, stratify=Y)

DAE_model = tf.keras.models.load_model("DAE_1000_600_2_old_freeze.h5")
DAE_predY = DAE_model.predict_classes(testX)
DAE_predY = 1 - DAE_predY

FFN_model = tf.keras.models.load_model("ffn_old.h5")
FFN_predY = FFN_model.predict_classes(testX)
FFN_predY = 1 - FFN_predY

print("DAE")
c_matrix = confusion_matrix(y_true=testY, y_pred=DAE_predY)
TN = float(c_matrix[0][0])
FP = float(c_matrix[0][1])
FN = float(c_matrix[1][0])
TP = float(c_matrix[1][1])
accuracy = (TP+TN)/(TP+FN+TN+FP)
sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)
f1_score = 2*TP/(2*TP+FP+FN)
print("accuracy", accuracy)
print("sensitivity", sensitivity)
print("specificity", specificity)
print("f1_score", f1_score)

print("FFN")
c_matrix = confusion_matrix(y_true=testY, y_pred=FFN_predY)
TN = float(c_matrix[0][0])
FP = float(c_matrix[0][1])
FN = float(c_matrix[1][0])
TP = float(c_matrix[1][1])
accuracy = (TP+TN)/(TP+FN+TN+FP)
sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)
f1_score = 2*TP/(2*TP+FP+FN)
print("accuracy", accuracy)
print("sensitivity", sensitivity)
print("specificity", specificity)
print("f1_score", f1_score)
