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
learning_rate = 0.0001
batch_size = 64
epochs = 50000
test_percentage = 0.20
num_nodes = 600
early_stopping_count = 100

#Split data into training set and test set
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=test_percentage, stratify=Y)

DAE_1000_model = tf.keras.models.load_model("DAE_1000_old.h5")
DAE_600_model = tf.keras.models.load_model("DAE_600_old.h5")

model = tf.keras.models.Sequential([
	      tf.keras.layers.Dense(1000, input_dim=num_features, activation=tf.nn.relu),
	      tf.keras.layers.Dense(600, input_dim=1000, activation=tf.nn.relu),
              tf.keras.layers.Dense(2, input_dim=600, activation=tf.nn.softmax)
])
model.layers[0].set_weights(DAE_1000_model.layers[0].get_weights())
model.layers[1].set_weights(DAE_600_model.layers[0].get_weights())

model.summary()
print(model.layers[0].get_weights())
print(DAE_1000_model.layers[0].get_weights())
print(model.layers[1].get_weights())
print(DAE_600_model.layers[0].get_weights())

#Compile the model
optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#Fit the model
callbacks = [tf.keras.callbacks.ModelCheckpoint("DAE_1000_600_2_old.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)]
model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, validation_data=(testX, testY), verbose=1, callbacks=callbacks)
