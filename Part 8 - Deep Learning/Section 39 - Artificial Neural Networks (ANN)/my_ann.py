# -*- coding: utf-8 -*-

# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow==1.5

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset=pd.read_csv("Churn_Modelling.csv")
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_encoder_X_1=LabelEncoder()
X[:,1]=label_encoder_X_1.fit_transform(X[:,1])
label_encoder_X_2=LabelEncoder()
X[:,2]=label_encoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing keras libraries and packages
import keras # Keras library is used to build neural network
from keras.models import Sequential # Sequential class is used to create an artificial neural network
from keras.layers import Dense # Dense class is used to create layers of an artificial neural network
# Dense class is also used to assign random weight for edge between 0 and 1

#Initialising the ANN
classifier=Sequential()

# Adding the input layer and first hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
# output_dim is the no of nodes in hidden layer. Generally we take the "Average of no. of node in input layer and 
# output layer that is 6" to compute the node in hidden layer
# init means intialisation of weight for edge. Uniform refers that we are uniformely assigning the weight
# activation means that which type of activation function we are using to calculate activation value of a node
# Here we use rectifier function for hidden layer which is written as 'relu' and sigmoid activation function
# for output layer.
# input_dim is the no. of nodes in input layer i.e 11

# Adding second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu')) # No need to pass the parameter for input_dim
# because neural network knows the node in the first hidden layer.

# Adding the output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

# We have completely make our neural network model

# Now we need to compile the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# optimizer means which algorithm we are using the calculate the weight which will be optimal. Adam is an algorithm
# loss means the cost function.
# metrics means the criterion we use to get the best of best result. Here we are using accuracy criterion
# to get the best model. Usually what happen, when algorithm calculate the weight,after that time it uses
# accuracy criterion to check the model performance.

#Fitting the ANN to our training set
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)
# batch_size=10 means after 10 obseravation we update the weight of edge.There is no critera to choose batch size
# nb_epoch=100 means we will train our model 100 times.

# Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test) # return probability of customer to leave the bank
y_pred = (y_pred > 0.5) # We need to convert probability into 0 or 1 so we do this by
# if(y_pred>0.5) return true else return false  This is equivalent to (y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# here we get an accuracy of 86.15 %