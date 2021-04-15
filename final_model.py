import tensorflow as tf
import keras
from  keras.datasets import mnist
from sklearn import model_selection
from keras import models

mnist=tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

x_train = x_train.reshape([60000, 28, 28, 1])
x_test= x_test.reshape([10000, 28, 28, 1])

x_train=keras.utils.normalize(x_train,axis=1)
x_test=keras.utils.normalize(x_test,axis=1)


model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (5, 5), padding="same", input_shape=[28, 28, 1]))
model.add(tf.keras.layers.MaxPool2D((2,2)))
model.add(tf.keras.layers.Conv2D(64, (5, 5), padding="same"))
model.add(tf.keras.layers.MaxPool2D((2,2)))
    
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5)

val_loss,val_acc=model.evaluate(x_test,y_test)
print(val_loss)
print(val_acc)
model.save('epic')
new_model=tf.keras.models.load_model('epic')
predictions=new_model.predict(x_test)
print(predictions)

import numpy as np
import matplotlib.pyplot as plt
print(np.argmax(predictions[3]))
plt.imshow(x_test[3],cmap=plt.cm.binary)
plt.show()
plt.imshow(x_test[0],cmap=plt.cm.binary)

from sklearn.metrics import confusion_matrix
pre=[np.argmax(i) for i in predictions]
cm=tf.math.confusion_matrix(labels=y_test,predictions=pre)

import seaborn as sn
plt.figure(figsize=(10,10))
plt.xlabel("Predict")
plt.ylabel("True")
sn.heatmap(cm,annot=True,fmt='d')


 