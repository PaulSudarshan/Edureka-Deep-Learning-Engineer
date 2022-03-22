# -*- coding: utf-8 -*-
"""Emotion-Recognition.ipynb

### Installing Kaggle package to pull dataset from kaggle
"""

!pip install kaggle

! mkdir ~/.kaggle

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

"""### Downloading Emotion Recognition dataset"""

! kaggle datasets download deadskull7/fer2013

"""### Unzipping the dataset"""

! unzip fer2013.zip

"""### Importing Required Libraries"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os

"""### Reading the dataset in csv format"""

df = pd.read_csv('fer2013.csv')
print(df.shape)
df.head()

"""### Dropping un-necessary column"""

df.drop(['Usage'],axis=1,inplace=True)

"""### Visualing random samples from the dataset"""

n=16
plt.figure(figsize=(12,10))

label_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

for en,i in enumerate(random.sample(range(1, 500), 16)):
    temp = np.array(df['pixels'][i].split(' '),dtype='float32').reshape((48,48,1))
    label = df['emotion'][i]
    plt.subplot(4,4,en+1);plt.imshow(tf.squeeze(temp),cmap='gray');plt.axis('off');plt.title(label_dict[label]);

"""### Creating Pre-processing function"""

def preprocessing(row):
    return np.array(row.split(' '),dtype='float').reshape((48,48,1))

"""### Applying the pre-processing function elementwise to the pixel column using 'apply' method"""

df['pixels']=df['pixels'].apply(preprocessing)

"""### Separating Feature and target variables."""

X = list(df['pixels'])
y = list(df['emotion'])

"""### Normalizing the images"""

X = np.array(X,dtype='float')/255.0

"""### Train-test Splitting the dataset"""

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



"""### Initiating Train Image Data-generator and performing Data Augmentation"""

train_datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

validation_datagen = ImageDataGenerator(rescale=1./255)

"""### Applying Image Augmentation and creating generators"""

batch_size = 32


train_generator = train_datagen.flow(
                        X_train,
                        y_train,
                        batch_size=batch_size,
                        shuffle=True)

validation_generator = validation_datagen.flow(
                                X_test,
                                y_test, 
                                batch_size=1,
                                shuffle=True)

"""### Defining the CNN Model for Emotion Recognition"""

def prepare_model():
    model = Sequential()
    
    model.add(Conv2D(64,kernel_size=(5,5),activation='relu',input_shape=(48, 48, 1)))
    model.add(Conv2D(64,kernel_size=(5,5),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128,kernel_size=(5,5),activation='relu'))
    model.add(Conv2D(128,kernel_size=(5,5),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256,kernel_size=(3,3),activation='relu'))
    model.add(Conv2D(256,kernel_size=(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(7, activation='softmax'))
    
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
    return model

model = prepare_model()

"""### Training the Model"""

model.fit_generator(train_generator, validation_data=(X_test,y_test),
 steps_per_epoch=len(X_train)//batch_size,epochs=40)
# 10 + 40 + 40 + 40 + 40

"""### Mounting Drive to save Trained Model every 40th Epoch"""

from google.colab import drive
drive.mount('content/')

model.save('content/MyDrive/colab/emotion/')

model.save('content/MyDrive/colab/emotion/model_170ep.h5')

"""### Loading Trained Model"""

model = tf.keras.models.load_model("content/MyDrive/colab/emotion/")

"""# Model Inferencing"""

n=20
plt.figure(figsize=(16,10))

label_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

for en,i in enumerate(random.sample(range(1, 500), 16)):
    temp = X_test[i].reshape(1,48,48,1)
    label = y_test[i].argmax()
    pred = model.predict(temp).argmax()
    temp*=255
    temp = temp.reshape((48,48,1))
    plt.subplot(5,4,en+1);plt.imshow(tf.squeeze(temp),cmap='gray');plt.axis('off');plt.title('True:'+label_dict[label]+'| Predicted:'+label_dict[pred]);





