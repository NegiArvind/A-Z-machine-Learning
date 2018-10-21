# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loaing the dataset
dataset=pd.read_csv("Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values

# Since we don't know how many number of clusters can be there so we use elbow method to find the 
# number of cluster  for this datset
from sklearn.cluster import KMeans
wcss=[] # wcss is within cluster secondary square 
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",n_init=10,max_iter=300,random_state=0)
    # n_clusters is number of cluster in which our dataset will be divided
    # init is the intialization of centroid of clusters using "k-means++" method
    # n_init is the number of time the k-means algorith will be run with different centroids.The final results
    # will be the best output of n_init consecutive runs in terms of inertia.
    # max_iter is the maximum number of iterations of the k-means algorithm for a single run
    kmeans.fit(X)
    print(kmeans.inertia_)
    wcss.append(kmeans.inertia_) # According to me, i think inertia_ returns the minimum cost for cluster size i. 
plt.plot(range(1,11),wcss)
plt.title("Elbow method of find cluster number")
plt.xlabel("Number of cluster")
plt.ylabel("WCSS")
plt.show()

# after visualizing the graph we find that number of cluster will be 5 for this dataset

# Applying kmeans for our dataset with right clusters size i.e 5
kmeans=KMeans(n_clusters=5,init="k-means++",n_init=10,max_iter=300,random_state=0)
y_kmeans=kmeans.fit_predict(X) # fit_predict method will return the cluster in which the client belong either 1,2,3,4 or 5

print(X[y_kmeans==0,0]) # returns an array containing value from X(column 0) where y_kmeans equal to zero

# visualizing the clusters
# since in python indexing starts from 0 so our first cluster will be 0,second cluster will be 1...
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='cluster 1') # s is the size of dots in graph
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='cluster 2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='cluster 3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='cluster 4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='centeroid') # plotting centroid of each cluster 
plt.title("Clusters of client")
plt.xlabel("Annual income in k$")
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.show()
