import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

X = np.load("X_old.npy") #(1035, 34716)
X = np.nan_to_num(X)
Y = np.load("Y_old.npy") #(1035,)
Y = np.array([Y, -(Y-1)]).T #(1035, 2) one-hot

seed = 10
tf.set_random_seed(seed)
np.random.seed(seed)

#Parameters
num_features = 34716
learning_rate = 50
batch_size = 64
epochs = 50000
test_percentage = 0.20
num_nodes = 1000

#Split data into training set and test set
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=test_percentage, stratify=Y)

corrupted_trainX = np.copy(trainX)
for i in range(len(trainX)):
	indices = np.random.choice(np.arange(num_features), replace=False,size=int(num_features * 0.2))
	corrupted_trainX[i][indices] = 0

corrupted_testX = np.copy(testX)
for i in range(len(testX)):
	indices = np.random.choice(np.arange(num_features), replace=False,size=int(num_features * 0.2))
	corrupted_testX[i][indices] = 0

model = tf.keras.models.Sequential([
              tf.keras.layers.Dense(num_nodes, input_dim=num_features, activation=tf.nn.relu),
              tf.keras.layers.Dense(num_features, activation=tf.nn.relu)
])

model.summary()

#Compile the model
optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
model.compile(optimizer = optimizer, loss='mean_squared_error')

#Fit the model
callbacks = [tf.keras.callbacks.ModelCheckpoint("DAE_1000_old.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)]
resu = model.fit(corrupted_trainX, trainX, batch_size=batch_size, epochs=epochs, validation_data=(corrupted_testX, testX), verbose=0, callbacks=callbacks)

