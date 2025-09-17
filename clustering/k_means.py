import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import random

X, y = make_blobs(n_samples=500, n_features=2, centers=3, random_state=24)

# plotting the dataset
fig = plt.figure(0)
plt.grid(True)
plt.scatter(X[:,0], X[:,1])
plt.show()

class k_mean():

    def __init__(self, k=3):
        self.k = k
        self.centroids = None        

    @staticmethod
    def distance_(data, centroids):
        return np.sqrt(np.sum((centroids - data) ** 2, axis=1))


    def fit(self, X, max_iter=300, tol=1e-4):
            self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), 
                                               size=(self.k, X.shape[1]))
            for _ in range(max_iter):
                y = []

                for data_point in X:
                    distances = k_mean.distance_(data_point, self.centroids)
                    cluster_num = np.argmin(distances)
                    y.append(cluster_num)

                y = np.array(y)

                cluster_center = []

                for i in range(self.k):
                     indices = np.where(y == i)[0] 
                     if len(indices) == 0:
                        cluster_center.append(self.centroids[i])
                     else:
                        cluster_center.append(np.mean(X[indices], axis=0)) 
                
                cluster_center = np.array(cluster_center)

                if np.max(np.abs(self.centroids - cluster_center)) < tol:
                     break  
                else:
                     self.centroids = cluster_center
            

            return y







km = k_mean(k = 3)
labels = km.fit(X)

fig = plt.figure(0)
plt.grid(True)
plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(km.centroids[:, 0], km.centroids[:, 1],
            c="red", marker="*", s=200, label="Centroids")
plt.show()
