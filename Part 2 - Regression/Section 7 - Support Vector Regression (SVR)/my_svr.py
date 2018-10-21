# -*- coding: utf-8 -*-


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # choosing column 1(we are writing it like so that we will get matrics not vector)
y = dataset.iloc[:, 2] .values

## Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =1/3, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X= sc_X.fit_transform(X)
y= sc_y.fit_transform(y)

# applying svr regressor on our dataset 
# SVR Model itself not perform feature scaling so we have to scale it manually
from sklearn.svm import SVR
regressor=SVR(kernel='rbf') #kernel is specifies which type of regresssor you want to apply
# Either linear or polynomial. rbf is default
regressor.fit(X,y)


# Predicting a specific result
# we have to predict salary for 6.5 level .Since we have done feature scaling so we have 
# to feature scale 6.5 also.
# we also need to inverse transform our y_pred 
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]])))) # since we have fit sc_x onto X 
# so here we will transform it only


#visualizing the data set
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title("Truth or Bluffs(Polynomial Regression)")
plt.xlabel("Levels")
plt.ylabel("Salary")

