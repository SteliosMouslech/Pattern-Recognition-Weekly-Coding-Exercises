import skfuzzy as fuzz
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt



iris=load_iris()
x=iris.data
y=iris.target

cntr, u, u0, distance_matrix, jm, p, fpc = fuzz.cluster.cmeans(x.T, 3, 2, error=0.005, maxiter=1000, seed=1998)

cluster_membership = np.argmax(u,axis=0)
print(cluster_membership)

#change our labels for correct prediction testing

y_cmeans=np.zeros(cluster_membership.shape)
y_cmeans[cluster_membership == 2]=0
y_cmeans[cluster_membership == 0]=1
y_cmeans[cluster_membership == 1]=2

#calculate error
errors=(np.where(y_cmeans == y, 0, 1)).sum()
error_rate=errors/y.shape[0]
print("Error rate of Fuzzy c-means on Iris Dataset is ",error_rate)

#sse calculation
sse=0
for i in range(x.shape[0]):
    sse=sse + distance_matrix[cluster_membership[i],i]**2
print("SSE is ",sse)

#plot our results
fig, axs = plt.subplots(2, 3)

axs[0, 0].scatter(x[y_cmeans == 0, 0], x[y_cmeans == 0, 1], s = 50, c ='red', label ='Iris-setosa')
axs[0, 0].scatter(x[y_cmeans == 1, 0], x[y_cmeans == 1, 1], s = 50, c ='blue', label ='Iris-versicolour')
axs[0, 0].scatter(x[y_cmeans == 2, 0], x[y_cmeans == 2, 1], s = 50, c ='green', label ='Iris-virginica')
axs[0, 0].scatter(cntr[:, 0], cntr[:,1], s = 50, c = 'yellow', label = 'Centroids')
axs[0,0].set(xlabel="Sepal Length", ylabel="Sepal Width")
axs[0, 0].legend()

axs[0,1].scatter(x[y_cmeans == 0, 0], x[y_cmeans == 0, 2], s = 50, c ='red', label ='Iris-setosa')
axs[0, 1].scatter(x[y_cmeans == 1, 0], x[y_cmeans == 1, 2], s = 50, c ='blue', label ='Iris-versicolour')
axs[0, 1].scatter(x[y_cmeans == 2, 0], x[y_cmeans == 2, 2], s = 50, c ='green', label ='Iris-virginica')
axs[0, 1].scatter(cntr[:, 0], cntr[:,2], s = 70, c = 'yellow', label = 'Centroids')
axs[0,1].set(xlabel="Sepal Length", ylabel="Petal Length")
axs[0,1].legend()



axs[0, 2].scatter(x[y_cmeans == 0, 0], x[y_cmeans == 0, 3], s = 50, c ='red', label ='Iris-setosa')
axs[0, 2].scatter(x[y_cmeans == 1, 0], x[y_cmeans == 1, 3], s = 50, c ='blue', label ='Iris-versicolour')
axs[0, 2].scatter(x[y_cmeans == 2, 0], x[y_cmeans == 2, 3], s = 50, c ='green', label ='Iris-virginica')
axs[0, 2].scatter(cntr[:, 0], cntr[:,3], s = 70, c = 'yellow', label = 'Centroids')
axs[0,2].set(xlabel="Sepal Length", ylabel=" Petal Width ")
axs[0, 2].legend()


axs[1, 0].scatter(x[y_cmeans == 0, 1], x[y_cmeans == 0, 2], s = 50, c ='red', label ='Iris-setosa')
axs[1, 0].scatter(x[y_cmeans == 1, 1], x[y_cmeans == 1, 2], s = 50, c ='blue', label ='Iris-versicolour')
axs[1, 0].scatter(x[y_cmeans == 2, 1], x[y_cmeans == 2, 2], s = 50, c ='green', label ='Iris-virginica')
axs[1, 0].scatter(cntr[:, 1], cntr[:,2], s = 70, c = 'yellow', label = 'Centroids')
axs[1,0].set(xlabel="Sepal Width", ylabel=" Petal Length ")
axs[1, 0].legend()


axs[1, 1].scatter(x[y_cmeans == 0, 1], x[y_cmeans == 0, 3], s = 50, c ='red', label ='Iris-setosa')
axs[1, 1].scatter(x[y_cmeans == 1, 1], x[y_cmeans == 1, 3], s = 50, c ='blue', label ='Iris-versicolour')
axs[1, 1].scatter(x[y_cmeans == 2, 1], x[y_cmeans == 2, 3], s = 50, c ='green', label ='Iris-virginica')
axs[1, 1].scatter(cntr[:, 1], cntr[:,3], s = 70, c = 'yellow', label = 'Centroids')
axs[1,1].set(xlabel="Sepal Width", ylabel=" Petal Width  ")
axs[1, 1].legend()


axs[1, 2].scatter(x[y_cmeans == 0, 2], x[y_cmeans == 0, 3], s = 50, c ='red', label ='Iris-setosa')
axs[1, 2].scatter(x[y_cmeans == 1, 2], x[y_cmeans == 1, 3], s = 50, c ='blue', label ='Iris-versicolour')
axs[1, 2].scatter(x[y_cmeans == 2, 2], x[y_cmeans == 2, 3], s = 50, c ='green', label ='Iris-virginica')
axs[1, 2].scatter(cntr[:, 2], cntr[:,3], s = 70, c = 'yellow', label = 'Centroids')
axs[1, 2].set(xlabel="Petal Length", ylabel=" Petal Width ")
axs[1, 2].legend()
#plt.tight_layout()

fig.suptitle("Fuzzy C Means on Iris Dataset (SEED=1998), m=2",fontsize=16)


plt.show()
