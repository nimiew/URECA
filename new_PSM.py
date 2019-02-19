"""Create PSMs and PSM matrices including the summed PSM/PSM Matrix"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split

seed = 10
np.random.seed(seed)

X = np.load("X_old.npy")
X = np.nan_to_num(X)

Y = np.load("Y_old.npy")
Y = np.nan_to_num(Y)
Y = np.array([Y, -(Y-1)]).T

trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, stratify=Y)
num_autistic = 0
num_control = 0
#Find number of autistic and control patients
for i in range(len(testY)):
    if(testY[i][0] == 1):
        num_autistic += 1
    else:
        num_control += 1
#Determine q values for q_autistic and q_control
q_autistic = float(num_autistic)/(num_autistic+num_control)
q_control = float(num_control)/(num_autistic+num_control)
q = np.zeros((len(testY),1))

for i in range(len(testY)):
    if(testY[i][0] == 1):
        q[i][0] = q_autistic
    else:
        q[i][0] = q_control
		
num_Of_PSMs = 32 #Get the first n psms

with h5py.File('/data/jang047/eigenvectors.hdf5', 'r') as g:
    print('reading eigenvectors')
    eigenvectors = g['eigenvectors']
    eigenvectors = np.array(eigenvectors)

table = np.zeros([len(testX), 3])
q = q.reshape((-1,))

for i in range(num_Of_PSMs):
        table = np.zeros([len(testX), 3])
        v = eigenvectors[:,i]
        for j in range(len(testX)):
                xv= np.dot(testX[j].T, v)
                table[j][0] = xv
        table[:,1] = q
        table[:,2] = table[:,0] * table[:,1]
        table = pd.DataFrame(table)
        table.to_csv('table_'+str(i)+'.csv')

