import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

X = np.load("X.npy") #(823, 34716)
X = np.nan_to_num(X)
Y = np.load("Y.npy") #(823,)
Y = np.array([Y, -(Y-1)]).T #(823, 2) one-hot

seed = 10
tf.set_random_seed(seed)
np.random.seed(seed)

#Parameters
num_features = 34716
learning_rate = 0.0001
batch_size = 64
epochs = 50000
test_percentage = 0.20
num_hidden_l1 = [100, 200, 300]
num_hidden_l2 = [100, 200, 300]
num_hidden_l3 = [100, 200, 300]
dropout_rate_l1 = 0.5
dropout_rate_l2 = 0.5
dropout_rate_l3 = 0.5
l2_regularization = 0.00001
early_stopping_count = 300

#result is used for storing information about each layer, so we can print and save it to csv file later
result = [[],[],[],[],[],[],[],[]]

#Split data into training set and test set
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=test_percentage, stratify=Y)

#Iterate through the various conbinations of nodes
for i in range(len(num_hidden_l1)):
    for j in range(len(num_hidden_l2)):
        for k in range(len(num_hidden_l3)):
            #Define the model
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
            #Compile the model
            model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
            #callbacks used to employ early stopping
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_count)]
            #Fit the model
            resu = model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, validation_data=(testX,testY), verbose=0, callbacks=callbacks)
            #Store relevant information into result array
            result[0].append(num_hidden_l1[i])
            result[1].append(num_hidden_l2[j])
            result[2].append(num_hidden_l3[k])
            result[3].append(resu.history['loss'][-1])
            result[4].append(resu.history['categorical_accuracy'][-1])
            result[5].append(resu.history['val_loss'][-1])
            result[6].append(resu.history['val_categorical_accuracy'][-1])
            result[7].append(len(resu.history['loss']))
            print("train loss:%.10g, train accuracy:%.10g, test loss:%.10g, test accuracy:%.10g for layers:%d-%d-%d at epoch %d"
                  %(resu.history['loss'][-1], resu.history['categorical_accuracy'][-1], resu.history['val_loss'][-1],
                  resu.history['val_categorical_accuracy'][-1],num_hidden_l1[i], num_hidden_l2[j], num_hidden_l3[k], len(resu.history['loss'])))

#Convert results into pandas dataframe, and subsequently csv file
result = pd.DataFrame(result)
result = result.transpose()
result.columns = ['Layer_1','Layer_2', 'Layer_3', 'Train_loss', 'Train_acc', 'Val_loss', 'Val_acc', 'Epoch stopped']
result.set_index('Layer_1') 
result.to_csv('result.csv')
