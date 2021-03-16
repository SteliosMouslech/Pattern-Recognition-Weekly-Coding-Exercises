import numpy as np
from sklearn.datasets import load_iris
import Ex5_myFunctions as ex5


iris=load_iris()
X=iris.data
Y=iris.target
data = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))
binary_classes_neg=np.where(Y>0,-1,1)
b=np.where(Y>0,-1,1)
weights=np.linalg.pinv(data).dot(b)
print("Using MSE with LS, weights are: ",weights)

#calculate error
decision_values=data.dot(np.atleast_2d(weights).T)
missclassified=np.sum((decision_values.T*binary_classes_neg)<0)
error_rate=missclassified/X.shape[0]
print("Using MSE error is: ",error_rate)


binary_classes=np.where(Y>0,0,1)
weights_lms,updates=ex5.duda_LMS(X.T,binary_classes,200,0.0001,0.02)
print("LMS Weights: :",weights_lms)
#calculate error
decision_values=data.dot(np.atleast_2d(weights_lms).T)
missclassified=np.sum((decision_values.T*binary_classes_neg)<0)
error_rate=missclassified/X.shape[0]
print("Using LMS error is: ",error_rate)
