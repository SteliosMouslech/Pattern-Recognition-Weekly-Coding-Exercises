import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron




X=np.array([[-2,1],[1,2],[0,0],[-2,-1],[2,0],[2,1]])
Y=np.array([1,1,0,0,0,0])

model = Perceptron(tol=1e-3, random_state=0)
model.fit(X,Y)
print('weights: ',model.coef_,"and bias",model.intercept_)
print("Single Perceptron Accuracy",model.score(X,Y))


x2= lambda x1:(-model.coef_[0][0]*x1-model.intercept_[0])/model.coef_[0][1]

#using our solution
weights=[[-1,+3]]
bias=[-1]
x2_my= lambda x1:(-weights[0][0]*x1-bias[0])/weights[0][1]
x1=np.linspace(-5,4,num=200)
fig = plt.figure()
ax = fig.add_subplot(111)
class1=Y==0
class2=Y==1
ax.scatter(X[class1,0],X[class1,1],c='r',marker='x')
ax.scatter(X[class2,0],X[class2,1],c='b',marker='o')
ax.plot(x1,x2(x1),label='sklearn Perceptron')
ax.plot(x1,x2_my(x1),label='By hand solution',c='g')
ax.legend()
# Move left y-axis and bottim x-axis to centre, passing through (0,0)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.title.set_text("Single Perceptron")
plt.show()