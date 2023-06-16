# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 20:52:52 2022

@author: user
"""

import matplotlib.pyplot as plt
# import nest_asyncio
import cv2 

import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from sklearn.model_selection import train_test_split
import numpy as np
from glob import glob
# import tensorflow_federated as tff
import os
from PIL import Image
import seaborn as sns


infected = os.listdir('C:/Users/user/Downloads/malaria_cell/cell_images/abc/Parasitized') 
uninfected = os.listdir('C:/Users/user/Downloads/malaria_cell/cell_images/abc/Uninfected')
data = []
labels = []

# print("****************** infected ***********************")
for i in infected:
    try:
        # print("reading image")
        image = cv2.imread('C:/Users/user/Downloads/malaria_cell/cell_images/abc/Parasitized/'+i)
        image_array = Image.fromarray(image , 'RGB')
        resize_img = image_array.resize((50 , 50))
        rotated45 = resize_img.rotate(45)
        rotated75 = resize_img.rotate(75)
        blur = cv2.blur(np.array(resize_img) ,(10,10))
        # print("read data")
        data.append(np.array(resize_img))
        data.append(np.array(rotated45))
        data.append(np.array(rotated75))
        data.append(np.array(blur))
        # print("labels")
        labels.append(1)
        labels.append(1)
        labels.append(1)
        labels.append(1)
        # print("end")
    
    except AttributeError:
        print('')

# print("****************** uninfected ***********************")
for u in uninfected:
    try:
        # print("read image")
        image = cv2.imread('C:/Users/user/Downloads/malaria_cell/cell_images/abc/Uninfected/'+u)
        image_array = Image.fromarray(image , 'RGB')
        resize_img = image_array.resize((50 , 50))
        rotated45 = resize_img.rotate(45)
        rotated75 = resize_img.rotate(75)
        # print("read data")
        data.append(np.array(resize_img))
        data.append(np.array(rotated45))
        data.append(np.array(rotated75))
        # print("read labels")
        labels.append(0)
        labels.append(0)
        labels.append(0)
        # print("end")
    
    except AttributeError:
        print('')

cells = np.asarray(data)
labels = np.asarray(labels)

# plt.figure(1 , figsize = (15 , 9))
# n = 0 
# for i in range(49):
#     n += 1 
#     r = np.random.randint(0 , cells.shape[0] , 1)
#     plt.subplot(7 , 7 , n)
#     plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
#     plt.imshow(cells[r[0]])
#     plt.title('{} : {}'.format('Infected' if labels[r[0]] == 1 else 'Unifected' ,
#                                labels[r[0]]) )
#     plt.xticks([]) , plt.yticks([])

# plt.show()

# plt.figure(1, figsize = (15 , 7))
# plt.subplot(1 , 2 , 1)
# plt.imshow(cells[0])
# plt.title('Infected Cell')
# plt.xticks([]) , plt.yticks([])

# plt.subplot(1 , 2 , 2)
# plt.imshow(cells[60000])
# plt.title('Uninfected Cell')
# plt.xticks([]) , plt.yticks([])

# plt.show()

cells = cells.astype(np.float32)
labels = labels.astype(np.int32)
cells = cells/255


x_train , x , y_train , y = train_test_split(cells , labels , 
                                            test_size = 0.2 ,
                                            random_state = 42)

eval_x , x_test , eval_y , y_test = train_test_split(x , y , 
                                                    test_size = 0.5 , 
                                                    random_state = 42)

x_train = x_train[:len(x_train)//3]
y_train = y_train[:len(y_train)//3]
# plt.figure(1 , figsize = (15 ,5))
# n = 0 
# for z , j in zip([train_y , eval_y , test_y] , ['train labels','eval labels','test labels']):
#     n += 1
#     plt.subplot(1 , 3  , n)
#     sns.countplot(x = z )
#     plt.title(j)
# plt.show()

# Load and compile Keras model
# vgg = VGG19(input_shape= (50,50,3), weights='imagenet', include_top=False)
# model=Sequential()
# model.add(vgg)
# model.add(Flatten())
# model.add(Dense(1024,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1,activation='sigmoid'))
# model.summary()
# model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

# Load and compile Keras model
model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),input_shape=(50,50,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
    
])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='binary_crossentropy',metrics=['acc'])

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]

model_history = model.fit(
    x_train,y_train,
    steps_per_epoch=50,
    validation_data=(eval_x,eval_y),
    validation_steps=100, 
    epochs = 10 , callbacks=my_callbacks)