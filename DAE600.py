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
learning_rate = 0.01
batch_size = 64
epochs = 50000
test_percentage = 0.20
num_nodes = 600

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

DAE_1000_model = tf.keras.models.load_model("DAE_1000_old.h5")
DAE_1000_model.summary()
intermediate_layer_model = tf.keras.models.Model(inputs=DAE_1000_model.input, outputs=DAE_1000_model.get_layer("dense").output)
trainX_for_600 = intermediate_layer_model.predict(corrupted_trainX)
testX_for_600 = intermediate_layer_model.predict(corrupted_testX)
print("trainX_for_600 shape is", trainX_for_600.shape)


corrupted_trainX_for_600 = np.copy(trainX_for_600)
for i in range(len(trainX_for_600)):
	indices = np.random.choice(np.arange(1000), replace=False,size=int(1000 * 0.3))
	corrupted_trainX_for_600[i][indices] = 0

corrupted_testX_for_600 = np.copy(testX_for_600)
for i in range(len(testX_for_600)):
	indices = np.random.choice(np.arange(1000), replace=False,size=int(1000 * 0.3))
	corrupted_testX_for_600[i][indices] = 0

model = tf.keras.models.Sequential([
              tf.keras.layers.Dense(num_nodes, input_dim=1000, activation=tf.nn.relu),
              tf.keras.layers.Dense(1000, activation=tf.nn.relu)
])

model.summary()

#Compile the model
optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
model.compile(optimizer = optimizer, loss='mean_squared_error')

#Fit the model
callbacks = [tf.keras.callbacks.ModelCheckpoint("DAE_600_old.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)]
resu = model.fit(corrupted_trainX_for_600, trainX_for_600, batch_size=batch_size, epochs=epochs, validation_data=(corrupted_testX_for_600, testX_for_600), verbose=1, callbacks=callbacks)


