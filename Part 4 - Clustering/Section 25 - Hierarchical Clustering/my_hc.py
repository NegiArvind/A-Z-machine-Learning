# -*- coding: utf-8 -*-

# Hierarichal Clustring

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset=pd.read_csv("Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values

# Drwing dendogram
import scipy.cluster.hierarchy as sch # this library contains all the tools used for hierarchial clustring and build dendogram
dendogram=sch.dendrogram(sch.linkage(X,method='ward')) # sch.linkage is a method need two parameter as an argument
# one is the array for which we have to make dendogram and second one is the method used to choose the clusters
# method='ward' .here "ward" method means that we use variance method to choose two closest cluster.
# Thos clusters having minimum variance will be cluster together.
plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Euclidian Distance")
plt.show()
 
#After visualizing Dendogram diagram, we caluclate the no. of cluster for our dataset i.e 5
# Fitting Hierarchial Clustering to our dataset
from sklearn.cluster import AgglomerativeClustering 
hc=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
y_hc=hc.fit_predict(X)

# Visualizing the content
# since in python indexing starts from 0 so our first cluster will be 0,second cluster will be 1...
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='red',label='cluster 1') # s is the size of dots in graph
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='blue',label='cluster 2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label='cluster 3')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='cyan',label='cluster 4')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='magenta',label='cluster 5') 
plt.title("Clusters of client")
plt.xlabel("Annual income in k$")
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.show()
