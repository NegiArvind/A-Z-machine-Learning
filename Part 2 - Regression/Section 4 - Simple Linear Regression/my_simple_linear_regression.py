 # -*- coding: utf-8 -*-
# single line comment = ctrl+1
# multiple line comment=ctrl+4

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1] .values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =1/3, random_state = 0)


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression() # LinearRegression is a class
regressor.fit(X_train,y_train) # it will fit the regressor object to the training set
# Means using this regressor object we will train our model

#Predicting the test result on test set
y_pred=regressor.predict(X_test)
# diff=y_test-y_pred

# Plotting the graph of training set result(Visualization)
plt.scatter(X_train,y_train,color='red') # it will make the points of red color on graph
y_trainPred=regressor.predict(X_train) 
plt.plot(X_train,y_trainPred,color='blue') # draw a line with predicted value of Y
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Year of Experience")
plt.ylabel("Salary")

#Plotting the graph of testing set result(Visualization)
plt.scatter(X_test,y_test,color='red') # it will make the points of red color on graph 
plt.plot(X_train,y_trainPred,color='blue') # draw a line with predicted value of Y
plt.title("Salary vs Experience (Testing set)")
plt.xlabel("Year of Experience")
plt.ylabel("Salary")


