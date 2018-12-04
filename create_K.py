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
print(q.shape)
#Load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
#Load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

#Find K
session = k.get_session()
session.run(tf.global_variables_initializer())
y_val_d_evaluated = session.run(tf.gradients(model.output, model.input), feed_dict={model.input: testX})
y_val_d_evaluated = np.array(y_val_d_evaluated)
r_T = y_val_d_evaluated.reshape(165,34716)
r = r_T.transpose()
r_T = r_T * q
result = np.matmul(r,r_T)

#Save K
with h5py.File('/data/jang047/K.hdf5', 'w') as f:
    dset = f.create_dataset("K", data=result)
