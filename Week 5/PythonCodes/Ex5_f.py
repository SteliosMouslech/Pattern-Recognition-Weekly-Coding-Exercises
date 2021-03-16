import numpy as np
from sklearn.datasets import load_iris
import Ex5_myFunctions as ex5
import matplotlib
import matplotlib.pyplot as plt


iris=load_iris()
X=iris.data
Y=iris.target

data=X.T
weights,bias,t=ex5.mperceptron(data,Y,1000)
fullWeights=np.vstack((weights,bias.T))
print("full Weights after 1000 iters for 3 classes\n",fullWeights)
dataAug = np.vstack((data, np.ones((1, data.shape[1]), dtype=data.dtype)))
test=np.dot(fullWeights.T,dataAug)
classifications=np.argmax(test.T,axis=1)
errors=np.sum(np.where(Y!=classifications,1,0))
print("Error Rate for multiclass classification Using Kesler for 1000 iters: ",errors/X.shape[0])
weights,bias,t=ex5.mperceptron(data,Y,10000)
fullWeights=np.vstack((weights,bias.T))
print("full Weights after 10000 iters for 3 classes\n",fullWeights)

dataAug = np.vstack((data, np.ones((1, data.shape[1]), dtype=data.dtype)))
test=np.dot(fullWeights.T,dataAug)
classifications=np.argmax(test.T,axis=1)
errors=np.sum(np.where(Y!=classifications,1,0))
print("Error Rate for multiclass classification Using Kesler for 10000 iters: ",errors/X.shape[0])