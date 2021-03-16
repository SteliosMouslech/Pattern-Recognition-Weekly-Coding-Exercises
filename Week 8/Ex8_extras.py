import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import skfuzzy as fuzz


iris=load_iris()
x=iris.data
y=iris.target
testK=[1,2,3,4,5,6,7,8,9,10]
SSEs=[]
for k in testK:
    kmeans = KMeans(n_clusters = k, init = 'random', max_iter = 500, n_init = 10, random_state = 0).fit(x)
    SSEs.append(kmeans.inertia_)




plt.plot(testK,SSEs)
plt.xticks(testK)
plt.xlabel("Number of Clusters")
plt.ylabel=("SSE")
plt.title("Sum Of Squared Error per K  (K-means)")
plt.show()

SSEs_fuzz=[]
for k in testK:
    cntr, u, u0, distance_matrix, jm, p, fpc = fuzz.cluster.cmeans(x.T, k, 2, error=0.005, maxiter=1000, seed=1998)
    cluster_membership = np.argmax(u,axis=0)
    sse=0
    for i in range(x.shape[0]):
        sse=sse + distance_matrix[cluster_membership[i],i]**2
    SSEs_fuzz.append(sse)

plt.plot(testK,SSEs_fuzz)
plt.xticks(testK)
plt.xlabel("Number of Clusters")
plt.ylabel=("SSE")
plt.title("Sum Of Squared Error per K (Fuzzy C means) ")
plt.show()