import numpy as np
from sklearn.datasets import load_iris
import Ex5_myFunctions as ex5


iris=load_iris()
X=iris.data
Y=iris.target
data = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))
#get w2 and w3 data
newData=data[50:150,:]
newLabels=Y[50:150]
binary_classes_neg=np.where(newLabels>1,1,-1)
b=np.where(newLabels>1,1,-1)
weights=np.linalg.pinv(newData).dot(b)
print("Using MSE with LS, weights are: ",weights)
decision_values=newData.dot(np.atleast_2d(weights).T)
missclassified=np.sum((decision_values.T*binary_classes_neg)<0)
error_rate=missclassified/newData.shape[0]

print("Using MSE error is: ",error_rate)




weights_Ho,b=ex5.duda_Ho_Kashyap(newData.T,binary_classes_neg,0,100000,0.001,0.0001)

test=weights_Ho.T.dot(newData.T)
missclassified=np.where(test>0,1,0)
k=np.squeeze(2*missclassified-1)
error_rate=np.sum(np.where(k!=binary_classes_neg,1,0))/newData.shape[0]
print("Using Ho Kashyap , weights are: ",weights_Ho.T)
print("Using Ηο Kashyap error is: ",error_rate)
