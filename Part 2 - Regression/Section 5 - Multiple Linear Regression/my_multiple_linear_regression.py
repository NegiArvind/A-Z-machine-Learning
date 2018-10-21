# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Encoding categorical data
# Encoding the Independent Variable
# LabelEncoder is used to just convert string into number so that we perform calculation
# machine learning model will think that 1 has higher priority than 0 and 2 has higher than 1
# so to remove this priorityness we use OnHotEncoder class 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding dummy variable trap
# if there are 3 dummy variable then we have to take only 2 dummy variable for our model
X=X[:,1:] # X contain all the column 1 afterwards


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# fiiting multiple linear regression onto our model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting profit
y_pred=regressor.predict(np.array(X_train))

# We cannot plot graph because there are too many independent variable
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,y_pred,color='blue')
plt.title("Multiple Linear Regression graph")
plt.show()

#Making optimal model using backward selection
# Steps in backward elimination
#step 1-select a significance level to stay in the model (e.g=0.05)
#step 2-fit the model with all possible predictor
#step 3-Consider the predictor with the highest p value. if p>sl then go to step 4 else model
# is ready.
# step 4-Remove the predictor
# step 5-Fit model without this predictor goto step3 

 # we need to multiply x0 with b1
 # y=b0*x0+b1*x1+b2*x2
 # satsmodels does not take constant value b0 if we will not multiply it with x0 
 # so we have to multiply it with x0. value of x0 will be 1
 # appending array of 1 at the beginning of  x
import statsmodels.formula.api as sm 
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit() # fitting model with OLS class
regressor_OLS.summary() # it will give us the all the details like p value

# p value of x2 is 0.990 which is higher then 0.05 so we will remove it
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit() # fitting model with OLS class
regressor_OLS.summary() # it will give us the all the details like p value

# p value of x4 is 0.602 which is higher then 0.05 so we will remove it
X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit() # fitting model with OLS class
regressor_OLS.summary() # it will give us all the details like p value of any independent variable

# p value of x5 is 0.060 which is higher then 0.05 so we will remove it
X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit() # fitting model with OLS class
regressor_OLS.summary() # it will give us all the details like p value,adjusted R squared value of any independent variable

# it means column 3 which is R&D spend predict perfect result.

regressor.fit(X_opt,y)
y_predict_OLS=regressor.predict(X_opt)
