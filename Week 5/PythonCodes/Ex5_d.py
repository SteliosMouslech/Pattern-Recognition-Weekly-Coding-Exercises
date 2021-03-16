import numpy as np
from sklearn.datasets import load_iris
import Ex5_myFunctions as ex5


iris=load_iris()
X=iris.data
Y=iris.target

data = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))
weights=[]
for i in range(3):
    b=np.where(Y==i,1,-1)
    weights.append(np.linalg.pinv(data).dot(b))
    print("Using MSE with LS for class,",i,", weights are: ",weights[i])

decision1=data.dot(np.atleast_2d(weights[0]).T)
decision2=data.dot(np.atleast_2d(weights[1]).T)
decision3=data.dot(np.atleast_2d(weights[2]).T)
decisionsGrouped=np.hstack((decision1,decision2,decision3))
classifications=np.argmax(decisionsGrouped,axis=1)
errors=np.sum(np.where(Y!=classifications,1,0))
print("Error Rate for multiclass classification: ",errors/X.shape[0])


