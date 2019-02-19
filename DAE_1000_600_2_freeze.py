import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

X = np.load("X.npy")
X = np.nan_to_num(X)
Y = np.load("Y.npy")
Y = np.array([Y, -(Y-1)]).T

seed = 10
tf.set_random_seed(seed)
np.random.seed(seed)

#Parameters
num_features = 34716
learning_rate = 0.00001
batch_size = 64
epochs = 100000
test_percentage = 0.20
num_nodes = 600
early_stopping_count = 100

#Split data into training set and test set
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=test_percentage, stratify=Y)

DAE_1000_model = tf.keras.models.load_model("DAE_1000.h5")
_600_layer_output_model = tf.keras.models.Model(inputs=DAE_1000_model.input, outputs=DAE_1000_model.get_layer("dense").output)
trainX_for_600 = _600_layer_output_model.predict(trainX)
testX_for_600 = _600_layer_output_model.predict(testX)

DAE_600_model = tf.keras.models.load_model("DAE_600.h5")
_2_layer_output_model = tf.keras.models.Model(inputs=DAE_600_model.input, outputs=DAE_600_model.get_layer("dense").output)
trainX_for_2 = _2_layer_output_model.predict(trainX_for_600)
testX_for_2 = _2_layer_output_model.predict(testX_for_600)

DAE_2_model = tf.keras.models.Sequential([
              tf.keras.layers.Dense(2, input_dim=600, activation=tf.nn.softmax),
])

#Compile the model
optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
DAE_2_model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#Fit the model
callbacks = [tf.keras.callbacks.ModelCheckpoint("DAE_2.h5", monitor='val_categorical_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)]
DAE_2_model.fit(trainX_for_2, trainY, batch_size=batch_size, epochs=epochs, validation_data=(testX_for_2, testY), verbose=1, callbacks=callbacks)

#Restore best weights
DAE_2_model = tf.keras.models.load_model("DAE_2.h5")

#Combine the models
full_model = tf.keras.models.Sequential([
	      tf.keras.layers.Dense(1000, input_dim=num_features, activation=tf.nn.relu),
	      tf.keras.layers.Dense(600, input_dim=1000, activation=tf.nn.relu),
          tf.keras.layers.Dense(2, input_dim=600, activation=tf.nn.softmax)
])
full_model.layers[0].set_weights(DAE_1000_model.layers[0].get_weights())
full_model.layers[1].set_weights(DAE_600_model.layers[0].get_weights())
full_model.layers[2].set_weights(DAE_2_model.layers[0].get_weights())
full_model.save("DAE_1000_600_2_freeze.h5")
