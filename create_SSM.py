"""Create SSM and SSM matrix"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import os
from keras import backend as k
import h5py

X = np.load("X.npy") #(823, 34716)
X = np.nan_to_num(X)
Y = np.load("Y.npy") #(823,)
Y = np.array([Y, -(Y-1)]).T #(823, 2) one-hot

seed = 10
tf.set_random_seed(seed)
np.random.seed(seed)

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
q_autistic = num_autistic/(num_autistic+num_control)
q_control = num_control/(num_autistic+num_control)
q = np.zeros((len(testY),1))
for i in range(len(testY)):
    if(testY[i][0] == 1):
        q[i][0] = q_autistic
    else:
        q[i][0] = q_control

#Load json and create model
with open('model.json', 'r') as f:
    model = model_from_json(f.read())
# Load weights into the new model
model.load_weights('model.h5')

#Find SSM
session = k.get_session()
session.run(tf.global_variables_initializer())
y_val_d_evaluated = session.run(tf.gradients(model.output, model.input), feed_dict={model.input: testX})
y_val_d_evaluated = np.array(y_val_d_evaluated)
SSM = y_val_d_evaluated.reshape(165,34716)
SSM = np.square(SSM) #Squares each value individually
SSM = SSM * q #Broadcasting is done here
SSM = np.sum(SSM, axis=0) #Sums in the vertical direction

#Save SSM
with h5py.File('SSM.hdf5', 'w') as f:
    dset = f.create_dataset("SSM", data=SSM)
    SSM = pd.DataFrame(SSM)
    SSM.to_csv("SSM.csv")

#Open SSM and recreate 264x264 matrix
with h5py.File('SSM.hdf5', 'r') as g:
    SSM = g['SSM']
    SSM = np.array(SSM)
    matrix = np.ndarray(shape=(264,264), dtype=float) #matrix stores final desired result
    idx = np.triu_indices(264, k=1) #Get indices of upper right triangle of matrix, excluding the diagonal
    matrix[:] = 0 #Set all values to 0
    matrix[idx] = SSM #Place ssm values into upper right triangle of matrix, using the indices defined above
    idx = np.tril_indices(264, k=-1) #Get indices of lower left triangle, excluding the diagonal
    matrix[idx] = matrix.T[idx] #Copy the mirror image of upper triangle to lower triangle

#Save SSM 264x264 matrix
with h5py.File('SSM_matrix.hdf5', 'w') as h:
    dset = h.create_dataset("SSM_matrix", data=matrix)
    matrix = pd.DataFrame(matrix)
    matrix.to_csv('SSM_matrix.csv')
