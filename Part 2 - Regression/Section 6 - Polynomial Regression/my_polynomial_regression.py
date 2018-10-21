# -*- coding: utf-8 -*-

#Polynomial Regression


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
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting linear regression to our set
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#fittting polynomial regression to our set
from sklearn.preprocessing import PolynomialFeatures
# PolynomialFeatures give us some tool by which we can transform x into x_poly(contains degree)
poly_reg=PolynomialFeatures(4) # 4 represent polynomial is of degree 4
X_poly=poly_reg.fit_transform(X) # fitting x into poly_reg and then transforming it 
# X_poly contains 4 column. values are x,x^2,x^3,x^4
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)

#visualising the linear regression result
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title("Truth or Bluffs(Linear Regression)")
plt.xlabel("Levels")
plt.ylabel("Salary")

# visulising the polynomial regression result
#x_grid=np.arange(min(X),max(X),0.1)  
#X_grid=np.reshape(len(x_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(X_poly),color='blue')
plt.title("Truth or Bluffs(Polynomial Regression)")
plt.xlabel("Levels")
plt.ylabel("Salary")