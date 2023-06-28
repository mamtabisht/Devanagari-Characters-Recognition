
import numpy as np
import os
import time
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
#from keras.layers import Dense, Activation, Flatten
#from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from keras.layers import Conv2D, MaxPool2D
from keras.models import Sequential, load_model
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten, Input
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# Loading the training data
#PATH = os.getcwd()
# Define data path
#data_path = PATH + '/data'

data_path ='D:/1Dataset/DevanagariHandwrittenCharacterDataset/Train'
data_dir_list = os.listdir(data_path)

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img
		img = image.load_img(img_path, target_size=(32, 32))
      
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		#x = x/255
		print('Input image shape:', x.shape)
		img_data_list.append(x)

img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)

# Define the number of classes
num_classes = 46
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:1700]=0
labels[1700:3400]=1
labels[3400:5100]=2
labels[5100:6800]=3
labels[6800:8500]=4
labels[8500:10200]=5
labels[10200:11900]=6
labels[11900:13600]=7
labels[13600:15300]=8
labels[15300:17000]=9
labels[17000:18700]=10
labels[18700:20400]=11
labels[20400:22100]=12
labels[22100:23800]=13
labels[23800:25500]=14
labels[25500:27200]=15
labels[27200:28900]=16
labels[28900:30600]=17
labels[30600:32300]=18
labels[32300:34000]=19
labels[34000:35700]=20
labels[35700:37400]=21
labels[37400:39100]=22
labels[39100:40800]=23
labels[40800:42500]=24
labels[42500:44200]=25
labels[44200:45900]=26
labels[45900:47600]=27
labels[47600:49300]=28
labels[49300:51000]=29
labels[51000:52700]=30
labels[52700:54400]=31
labels[54400:56100]=32
labels[56100:57800]=33
labels[57800:59500]=34
labels[59500:61200]=35
labels[61200:62900]=36
labels[62900:64600]=37
labels[64600:66300]=38
labels[66300:68000]=39
labels[68000:69700]=40
labels[69700:71400]=41
labels[71400:73100]=42
labels[73100:74800]=43
labels[74800:76500]=44
labels[76500:78200]=45


names = ['ka','kha','ga','gha','kna','cha','chha','ja','jha','yna','taamatar',
         'thaa','daa','dhaa','adna','tabala','tha','da','dha','na','pa','pha','ba','bha','ma',
         'yaw','ra','la','waw','motosaw','petchiryakha','patalosaw','ha','chhya','tra','gya','digit_0',
         'digit_1','digit_2','digit_3','digit_4','digit_5','digit_6','digit_7','digit_8','digit_9']



# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x_train,y_train = shuffle(img_data,Y, random_state=2)
# Split the dataset
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


#validation and Testing
data_path_test ='D:/DevanagariHandwrittenCharacterDataset/Test'
data_dir_list_test = os.listdir(data_path_test)

img_data_list_test=[]

for dataset_test in data_dir_list_test:
	img_list_test=os.listdir(data_path_test+'/'+ dataset_test)
	print ('Loaded the images of dataset_test-'+'{}\n'.format(dataset_test))
	for img in img_list_test:
		img_path_test = data_path_test + '/'+ dataset_test + '/'+ img
		img = image.load_img(img_path_test, target_size=(32, 32))
      
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		#x = x/255
		print('Input image shape:', x.shape)
		img_data_list_test.append(x)

img_data_test = np.array(img_data_list_test)
#img_data = img_data.astype('float32')
print (img_data_test.shape)
img_data_test=np.rollaxis(img_data_test,1,0)
print (img_data_test.shape)
img_data_test=img_data_test[0]
print (img_data_test.shape)
# Define the number of classes
num_classes = 46
num_of_samples = img_data_test.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:300]=0
labels[300:600]=1
labels[600:900]=2
labels[900:1200]=3
labels[1200:1500]=4
labels[1500:1800]=5
labels[1800:2100]=6
labels[2100:2400]=7
labels[2400:2700]=8
labels[2700:3000]=9
labels[3000:3300]=10
labels[3300:3600]=11
labels[3600:3900]=12
labels[3900:4200]=13
labels[4200:4500]=14
labels[4500:4800]=15
labels[4800:5100]=16
labels[5100:5400]=17
labels[5400:5700]=18
labels[5700:6000]=19
labels[6000:6300]=20
labels[6300:6600]=21
labels[6600:6900]=22
labels[6900:7200]=23
labels[7200:7500]=24
labels[7500:7800]=25
labels[7800:8100]=26
labels[8100:8400]=27
labels[8400:8700]=28
labels[8700:9000]=29
labels[9000:9300]=30
labels[9300:9600]=31
labels[9600:9900]=32
labels[9900:10200]=33
labels[10200:10500]=34
labels[10500:10800]=35
labels[10800:11100]=36
labels[11100:11400]=37
labels[11400:11700]=38
labels[11700:12000]=39
labels[12000:12300]=40
labels[12300:12600]=41
labels[12600:12900]=42
labels[12900:13200]=43
labels[13200:13500]=44
labels[13500:13800]=45


names = ['ka','kha','ga','gha','kna','cha','chha','ja','jha','yna','taamatar',
         'thaa','daa','dhaa','adna','tabala','tha','da','dha','na','pa','pha','ba','bha','ma',
         'yaw','ra','la','waw','motosaw','petchiryakha','patalosaw','ha','chhya','tra','gya','digit_0',
         'digit_1','digit_2','digit_3','digit_4','digit_5','digit_6','digit_7','digit_8','digit_9']



# convert class labels to on-hot encoding
Y_test = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data_test,Y_test, random_state=2)
# Split the dataset
x_val, x_test, y_val, y_test = train_test_split(x, y, test_size=0.5, random_state=2)


###########################################################################
n_classes = 46
img_height_rows = 32
img_width_cols = 32
im_shape = (img_height_rows, img_width_cols, 3)
#CNN Model - Sequential Modelling
cnn = Sequential()
kernelSize = (3, 3)
ip_activation = 'relu'
#ip_activation = 'sigmoid'
ip_conv_0 = Conv2D(filters=32, kernel_size=kernelSize, input_shape=im_shape, activation=ip_activation)
cnn.add(ip_conv_0)
# Add the next Convolutional+Activation layer
ip_conv_0_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
cnn.add(ip_conv_0_1)

# Add the Pooling layer
pool_0 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
cnn.add(pool_0)
ip_conv_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
cnn.add(ip_conv_1)
ip_conv_1_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
cnn.add(ip_conv_1_1)

pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
cnn.add(pool_1)

# Let's deactivate around 20% of neurons randomly for training
drop_layer_0 = Dropout(0.2)
cnn.add(drop_layer_0)

flat_layer_0 = Flatten()
cnn.add(Flatten())
# Now add the Dense layers
h_dense_0 = Dense(units=128, activation=ip_activation, kernel_initializer='uniform')
cnn.add(h_dense_0)
# Let's add one more before proceeding to the output layer
h_dense_1 = Dense(units=64, activation=ip_activation, kernel_initializer='uniform')
cnn.add(h_dense_1)

op_activation = 'softmax'
output_layer = Dense(units=n_classes, activation=op_activation, kernel_initializer='uniform')
cnn.add(output_layer)

#opt = 'adam'
opt = 'sgd'
loss = 'categorical_crossentropy'
metrics = ['accuracy']
# Compile the classifier using the configuration we want
cnn.compile(optimizer=opt, loss=loss, metrics=metrics)
print(cnn.summary())


filepath = "./weights/temp/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(
        filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max')

callbacks_list = [checkpoint]


history = cnn.fit(x_train, y_train,
                  batch_size=32, epochs=50,verbose=1,
                  validation_data=(x_val, y_val),callbacks=callbacks_list)

#Save model, load model and best saved weights
cnn.save('my_model.hdf5')
model=load_model('my_model.hdf5')
model.load_weights('weights-improvement-43-0.99.hdf5')

scores = model.evaluate(x_val, y_val, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


scores = model.evaluate(x_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

import matplotlib.pyplot as plt
# Accuracy
print(history)
fig1, ax_acc = plt.subplots()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model - Accuracy; train= 78200, val= 6900, test= 6900')
#plt.title('train= 78200, val= 11040, test= 2760')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()
#According to the accuracy curves, the training and validation curves clearly follow the same trend throughout. 
#This is not a case of overfitting or underfitting. 

# Loss
fig2, ax_loss = plt.subplots()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model- Loss')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()



#prediction
# Final evaluation of the model
scores = model.evaluate(x_test,y_test, verbose=1)

#print('Large CNN Error: %.2f%%' % (100-scores[1]*100))
print('Test loss:', scores[0])
print('Test Accuracy : %.2f%%' % (scores[1]*100))

#print('Test Loss:', scores[0])
#print('Test accuracy:', scores[1])



from sklearn.metrics import confusion_matrix
preds = model.predict(x_test)
predicts = np.argmax(preds, axis = 1)
Y_test_labels = np.argmax(y_test, axis =1)
cm = confusion_matrix(Y_test_labels, predicts)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


print('Actual_labels  |  Predicted_labels')
for i in range(20):
    print('      %3d      |     %3d' %(Y_test_labels[i], predicts[i]))
    
count=0
print('Actual_labels  |  Predicted_labels')
for i in range(num_of_samples):
    if (Y_test_labels[i]!=predicts[i]):
        print(i)
        count=count+1
        print('      %3d      |     %3d' %(Y_test_labels[i], predicts[i]) ) 
    
print(count)

#or other method
# Printing the confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = model.predict(x_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
#y_pred = model.predict_classes(X_test)
#print(y_pred)
target_names = ['ka','kha','ga','gha','kna','cha','chha','ja','jha','yna','taamatar',
         'thaa','daa','dhaa','adna','tabala','tha','da','dha','na','pa','pha','ba','bha','ma',
         'yaw','ra','la','waw','motosaw','petchiryakha','patalosaw','ha','chhya','tra','gya','digit_0',
         'digit_1','digit_2','digit_3','digit_4','digit_5','digit_6','digit_7','digit_8','digit_9']


print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))


# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
#plt.figure()
# Plot normalized confusion matrix
#plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
#                      title='Normalized confusion matrix')
#plt.figure()
plt.show()


