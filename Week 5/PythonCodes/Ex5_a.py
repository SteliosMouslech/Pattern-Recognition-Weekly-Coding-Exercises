from sklearn.datasets import load_iris
import numpy as np
import Ex5_myFunctions as ex5

iris=load_iris()
X=iris.data
Y=iris.target
class_labels_perceptron=np.where(Y>0,-1,1)
best_weights,errors,iters=ex5.duda_batch_Perceptron(X,class_labels_perceptron,0.09,20,np.array([1,1,1,1]))

print("Batch Perceptron ,withs weights: ",best_weights)
print("Number of missclassifications: ",errors[-1])
print("After ",iters," Iterations")

class_labels_perceptron2=np.where(Y>0,1,0)
weights_margin,iter,errors_perepoch=ex5.duda_batch_Perceptron_with_Margin(X.T,class_labels_perceptron2,0.2,0.02,40)

print("Batch perceptron with Margin with Weights: :",weights_margin)

#error calculation
data = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))
decision_values=data.dot(np.atleast_2d(weights_margin).T)
missclassified=np.sum((decision_values.T*class_labels_perceptron)>0) #>0 cause we make -data the data of class 0 in margin batch perceptron
error_rate=missclassified/X.shape[0]
print("Using Batch perceptron with Margin,error is: ",error_rate)



