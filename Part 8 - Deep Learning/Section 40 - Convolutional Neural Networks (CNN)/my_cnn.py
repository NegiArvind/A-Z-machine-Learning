# -*- coding: utf-8 -*-

# Convolutional neural network

# Importing the libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


image_width=100
image_height=100
batch_size=32
epoch=2
# Intialising the CNN
classifier=Sequential()



# Step 1 Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(image_width,image_height,3),activation='relu'))
# Here 32 is number of feature detector we are using in our cnn model.Generally we use 32 feature detector if we are executing
# our model on cpu
# 3,3 is the no.of rows and no. of vertex
# input_shape(64,64,3) means input_shape is a 3d matrix where 64,64 are the number of pixels in rows and column
# and 3 is the no. of channel(red,green,blue)
# relu means we are using rectified linear unit method to find the activation value.
# After rectifier the pixel matrix size decreases


#step 2 MaxPooling
classifier.add(MaxPooling2D(pool_size=(2,2))) 
# 2,2 is the number of rows and column
# after pooling the Matrix size decreases

# Adding second convolution layer
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding third convolution layer
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2))) 


#Step 3 Flattening
classifier.add(Flatten())
# it will convert the 2D matrix into vector matrix


# Step 4-Full connection
classifier.add(Dense(output_dim=128,activation='relu')) # Hidden layer or fully connected layer having 128 node
classifier.add(Dense(output_dim=1,activation='sigmoid')) #Output layer

# Compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the model to our dataset

# we will use keras here to process the dataset
# ImageDataGenerator class of Keras library give us the facility of augmentation.
# Augmentation means it itself create many images of original image after performing rotate,shear,
# stretch,flip etc operation.it train our model better and give correct result.
# it also helps the model from overfitting. 

from keras.preprocessing.image import ImageDataGenerator

train_datagenerator= ImageDataGenerator(rescale = 1./255, #scaling matrix value in between of 0-1
                                   shear_range = 0.2, # how much the image can be shear
                                   zoom_range = 0.2, # how much the  image can be zoom
                                   horizontal_flip = True)

test_datagenerator = ImageDataGenerator(rescale = 1./255)

training_set = train_datagenerator.flow_from_directory('dataset/training_set',
                                                 target_size = (image_width, image_height), # all images will be resized to 64x64
                                                 batch_size = batch_size, # 32 training image will go for forward pass/backward pass
                                                 class_mode = 'binary')
# we divide the training set into batches to reduce the memory usage.

test_set = test_datagenerator.flow_from_directory('dataset/test_set',
                                            target_size = (image_width, image_height),
                                            batch_size = batch_size,  
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = epoch,
                         validation_data = test_set,
                         nb_val_samples = 2000) # 8000 images present inside trainingset folder
 #here we also pass test set. and get the result of test set
 # 2000 images present inside test set folder

classifier.save('model.h5')