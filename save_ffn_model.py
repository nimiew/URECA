"""Save the chosen ffn model"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import os

X = np.load("X_old.npy")
X = np.nan_to_num(X)
Y = np.load("Y_old.npy")
Y = np.array([Y, -(Y-1)]).T

seed = 10
tf.set_random_seed(seed)
np.random.seed(seed)

#Parameters
num_features = 34716
learning_rate = 0.0001
batch_size = 64
epochs = 5000
test_percentage = 0.20
num_hidden_l1 = [400]
num_hidden_l2 = [400]
num_hidden_l3 = [500]
dropout_rate_l1 = 0.5
dropout_rate_l2 = 0.5
dropout_rate_l3 = 0.5
l2_regularization = 0.00001
early_stopping_count = 300

trainX, testX, trainY, testY = train_test_split(X, Y, test_size=test_percentage, stratify=Y)

for i in range(len(num_hidden_l1)):
    for j in range(len(num_hidden_l2)):
        for k in range(len(num_hidden_l3)):
            model = tf.keras.models.Sequential([
              tf.keras.layers.Dense(num_hidden_l1[i], input_dim=num_features, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)),
              tf.keras.layers.Dropout(1-dropout_rate_l1),
              tf.keras.layers.Dense(num_hidden_l2[j], activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)),
              tf.keras.layers.Dropout(1-dropout_rate_l2),
              tf.keras.layers.Dense(num_hidden_l3[k], activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)),
              tf.keras.layers.Dropout(1-dropout_rate_l3),
              tf.keras.layers.Dense(2, activation=tf.nn.softmax)
            ])
            optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)

            model.compile(optimizer = optimizer,
                          loss='categorical_crossentropy',
                          metrics=['categorical_accuracy'])
            callbacks = [tf.keras.callbacks.ModelCheckpoint("ffn_old.h5", monitor='val_categorical_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)]
            model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, validation_data=(testX, testY), verbose=1, callbacks=callbacks)


