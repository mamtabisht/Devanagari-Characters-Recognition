import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
#matplotlib inline
import tensorflow as tf
import keras
import glob
import cv2
import pickle, datetime

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers import LSTM, Input, TimeDistributed,Convolution2D,Activation
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import RMSprop, SGD
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import optimizers
from keras.preprocessing import sequence
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import load_model

from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Import the backend
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
import os

print(os.listdir("D:/1Dataset/DevanagariHandwrittenCharacterDataset"))

#Read the train & test Images and preprocessing
train_images = []
train_labels = [] 
for directory_path in glob.glob("D:/1Dataset/DevanagariHandwrittenCharacterDataset/Train/*"):
    label = directory_path.split("_")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (32,32))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)
train_images = np.array(train_images)
train_labels = np.array(train_labels)

label_to_id = {v:i for i,v in enumerate(np.unique(train_labels))}
id_to_label = {v: k for k, v in label_to_id.items()}
train_label_ids = np.array([label_to_id[x] for x in train_labels])

print(train_images.shape),print( train_label_ids.shape), print(train_labels.shape)

# test
test_images = []
test_labels = [] 
for directory_path in glob.glob("D:/1Dataset/DevanagariHandwrittenCharacterDataset/Test/*"):
    test_label = directory_path.split("_")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (32,32))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(test_label)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

test_label_ids = np.array([label_to_id[x] for x in test_labels])

print(test_images.shape), print(test_label_ids.shape)

x_train, y_train, x_test, y_test, N_CATEGORY =train_images,train_labels,test_images,test_labels,len(label_to_id)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, N_CATEGORY)

print(id_to_label)




model=load_model('my_model.hdf5')
model.summary()
model.load_weights('weights-improvement-43-0.99.hdf5')

out= model.get_layer('dense_8').output
cnnRF_model = Model(model.input, out)
cnnRF_model .summary()


for layer in cnnRF_model.layers[:]:
	layer.trainable = False
cnnRF_model.layers[3].trainable


#Find the Features for n number of train images and we will get n x 64
#This means we will get 64 features for each images.
i=0
features=np.zeros(shape=(x_train.shape[0],64))
for directory_path in glob.glob("D:/1Dataset/DevanagariHandwrittenCharacterDataset/Train/*"):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)    
        img = cv2.resize(img, (32,32))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.expand_dims(img, axis=0)
        FC_output = cnnRF_model.predict(img)
        features[i]=FC_output
        i+=1


#Save the features of the train images to use it in future.
np.save('features', features)


#Name the feature rows as f_0, f_1, f_2...
feature_col=[]
for i in range(64):
    feature_col.append("f_"+str(i))
    i+=1
    
#Create DataFrame with features and coloumn name
train_features=pd.DataFrame(data=features,columns=feature_col)
feature_col = np.array(feature_col)

train_class = list(np.unique(train_label_ids))
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_label_ids.shape)
print(train_class)    


#Feed the extracted features with the labels to RANDOM FOREST 
rf = RandomForestClassifier(n_estimators = 20, random_state = 42,max_features=4)

rf.fit(train_features, train_label_ids)

#Find the Features from Alexnet's FC layer for n number of test images and we will get n x 64
i=0
features_test=np.zeros(shape=(y_test.shape[0],64))
for directory_path in glob.glob("D:/1Dataset/DevanagariHandwrittenCharacterDataset/Test/*"):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)    
        img = cv2.resize(img, (32,32))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.expand_dims(img, axis=0)
        FC_output = cnnRF_model.predict(img)
        features_test[i]=FC_output
        i+=1
        

#Create DataFrame with features and coloumn name
test_features=pd.DataFrame(data=features_test,columns=feature_col)
feature_col = np.array(feature_col)

print('Test Features Shape:', test_features.shape)
print('Test Labels Shape:', test_label_ids.shape)        

#Feed the features of the test images to Random Forest Classifier to predict its class
predictions = rf.predict(test_features)


accuracy=accuracy_score(predictions , test_label_ids)
print('Accuracy:', accuracy*100, '%.')

