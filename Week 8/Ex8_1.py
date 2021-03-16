import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris=load_iris()
x=iris.data
y=iris.target

kmeans = KMeans(n_clusters = 3, init = 'random', max_iter = 500, n_init = 10, random_state = 0).fit(x)
y_kmeans_og = kmeans.labels_
print(y_kmeans_og)

#K-means labels are different from ours we change it for visual reasons only and error calculation since we know classes
y_kmeans=np.zeros(y_kmeans_og.shape)
y_kmeans[y_kmeans_og==2]=0
y_kmeans[y_kmeans_og==0]=1
y_kmeans[y_kmeans_og==1]=2

#calculate error
errors=(np.where(y_kmeans==y,0,1)).sum()
error_rate=errors/y.shape[0]
print("Error rate of K-means on Iris Dataset is ",error_rate)
print("SSE is ",kmeans.inertia_)

#plot our results
fig, axs = plt.subplots(2, 3)

axs[0, 0].scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Iris-setosa')
axs[0, 0].scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1,1], s = 50, c = 'blue', label = 'Iris-versicolour')
axs[0, 0].scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2,1], s = 50, c = 'green', label = 'Iris-virginica')
axs[0, 0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 50, c = 'yellow', label = 'Centroids')
axs[0,0].set(xlabel="Sepal Length", ylabel="Sepal Width")
axs[0, 0].legend()

axs[0,1].scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 2], s = 50, c = 'red', label = 'Iris-setosa')
axs[0, 1].scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1,2], s = 50, c = 'blue', label = 'Iris-versicolour')
axs[0, 1].scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2,2], s = 50, c = 'green', label = 'Iris-virginica')
axs[0, 1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,2], s = 70, c = 'yellow', label = 'Centroids')
axs[0,1].set(xlabel="Sepal Length", ylabel="Petal Length")
axs[0,1].legend()



axs[0, 2].scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 3], s = 50, c = 'red', label = 'Iris-setosa')
axs[0, 2].scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1,3], s = 50, c = 'blue', label = 'Iris-versicolour')
axs[0, 2].scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2,3], s = 50, c = 'green', label = 'Iris-virginica')
axs[0, 2].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,3], s = 70, c = 'yellow', label = 'Centroids')
axs[0,2].set(xlabel="Sepal Length", ylabel=" Petal Width ")
axs[0, 2].legend()


axs[1, 0].scatter(x[y_kmeans == 0, 1], x[y_kmeans == 0, 2], s = 50, c = 'red', label = 'Iris-setosa')
axs[1, 0].scatter(x[y_kmeans == 1, 1], x[y_kmeans == 1,2], s = 50, c = 'blue', label = 'Iris-versicolour')
axs[1, 0].scatter(x[y_kmeans == 2, 1], x[y_kmeans == 2,2], s = 50, c = 'green', label = 'Iris-virginica')
axs[1, 0].scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:,2], s = 70, c = 'yellow', label = 'Centroids')
axs[1,0].set(xlabel="Sepal Width", ylabel=" Petal Length ")
axs[1, 0].legend()


axs[1, 1].scatter(x[y_kmeans == 0, 1], x[y_kmeans == 0, 3], s = 50, c = 'red', label = 'Iris-setosa')
axs[1, 1].scatter(x[y_kmeans == 1, 1], x[y_kmeans == 1,3], s = 50, c = 'blue', label = 'Iris-versicolour')
axs[1, 1].scatter(x[y_kmeans == 2, 1], x[y_kmeans == 2,3], s = 50, c = 'green', label = 'Iris-virginica')
axs[1, 1].scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:,3], s = 70, c = 'yellow', label = 'Centroids')
axs[1,1].set(xlabel="Sepal Width", ylabel=" Petal Width  ")
axs[1, 1].legend()


axs[1, 2].scatter(x[y_kmeans == 0, 2], x[y_kmeans == 0, 3], s = 50, c = 'red', label = 'Iris-setosa')
axs[1, 2].scatter(x[y_kmeans == 1, 2], x[y_kmeans == 1,3], s = 50, c = 'blue', label = 'Iris-versicolour')
axs[1, 2].scatter(x[y_kmeans == 2, 2], x[y_kmeans == 2,3], s = 50, c = 'green', label = 'Iris-virginica')
axs[1, 2].scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:,3], s = 70, c = 'yellow', label = 'Centroids')
axs[1, 2].set(xlabel="Petal Length", ylabel=" Petal Width ")
axs[1, 2].legend()
#plt.tight_layout()

fig.suptitle("K-means on Iris Dataset (Random_state=0)",fontsize=16)



plt.show()