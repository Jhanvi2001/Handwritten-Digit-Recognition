# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 10:14:23 2021

@author: sjhan
"""

import tensorflow as tf
import keras
from  keras.datasets import mnist
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import seaborn as sn

(x_train,y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train=x_train/255
x_test=x_test/255

x_train=x_train.reshape(len(x_train),28*28)
x_test=x_test.reshape(len(x_test),28*28)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test)

import numpy as np
y=model.predict(x_test)
print(y)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
c=tf.math.confusion_matrix(labels=y_test,predictions=y)
plt.figure(figsize=(10,7))
sn.heatmap(c,annot=True,fmt='d')
plt.xlabel("Prediction")
plt.ylabel("True")
