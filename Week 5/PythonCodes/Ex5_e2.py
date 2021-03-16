import numpy as np
from sklearn.datasets import load_iris
import Ex5_myFunctions as ex5
import matplotlib
import matplotlib.pyplot as plt


iris=load_iris()
X=iris.data
Y=iris.target

data_dims123=X[:,1:4]

data = np.hstack((data_dims123, np.ones((data_dims123.shape[0], 1), dtype=X.dtype)))
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
print("Error Rate for multiclass classification: ",errors/data_dims123.shape[0])


class0=Y==0
class1=Y==1
class2=Y==2

z1 = lambda x,y: (-weights[0][3]-weights[0][0]*x -weights[0][1]*y) / weights[0][2]
z2 = lambda x,y: (-weights[1][3]-weights[1][0]*x -weights[1][1]*y) / weights[1][2]
z3 = lambda x,y: (-weights[2][3]-weights[2][0]*x -weights[2][1]*y) / weights[2][2]

tmp = np.linspace(0,10,30)
x,y = np.meshgrid(tmp,tmp)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_dims123[class0,0],data_dims123[class0,1],data_dims123[class0,2], c='r', marker='o',label='Iris Setosa')
ax.scatter(data_dims123[class1,0],data_dims123[class1,1],data_dims123[class1,2], c='b', marker='o',label='Iris Versicolour')
ax.scatter(data_dims123[class2,0],data_dims123[class2,1],data_dims123[class2,2], c='g', marker='o',label='Iris Virginica')
ax.plot_surface(x, y, z1(x,y),color='r')

ax.legend()
ax.set_title('Decision Boundary for class 1: Iris Setosa')
ax.set_xlabel('Sepal Width (m)')
ax.set_ylabel('Petal Length (m)')
ax.set_zlabel('Petal Width (m)')



fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')

ax2.plot_surface(x, y, z2(x,y),color='b')
ax2.scatter(data_dims123[class0,0],data_dims123[class0,1],data_dims123[class0,2], c='r', marker='o',label='Iris Setosa')
ax2.scatter(data_dims123[class1,0],data_dims123[class1,1],data_dims123[class1,2], c='b', marker='o',label='Iris Versicolour')
ax2.scatter(data_dims123[class2,0],data_dims123[class2,1],data_dims123[class2,2], c='g', marker='o',label='Iris Virginica')

ax2.legend()
ax2.set_title('Decision Boundary for class 2: Iris Versicolour')
ax2.set_xlabel('Sepal Width (m)')
ax2.set_ylabel('Petal Length (m)')
ax2.set_zlabel('Petal Width (m)')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')

ax3.plot_surface(x, y, z3(x,y),color='g')
ax3.scatter(data_dims123[class0,0],data_dims123[class0,1],data_dims123[class0,2], c='r', marker='o',label='Iris Setosa')
ax3.scatter(data_dims123[class1,0],data_dims123[class1,1],data_dims123[class1,2], c='b', marker='o',label='Iris Versicolour')
ax3.scatter(data_dims123[class2,0],data_dims123[class2,1],data_dims123[class2,2], c='g', marker='o',label='Iris Virginica')

ax3.legend()
ax3.set_title('Decision Boundary for class 3: Iris Versicolour')
ax3.set_xlabel('Sepal Width (m)')
ax3.set_ylabel('Petal Length (m)')
ax3.set_zlabel('Petal Width (m)')

fig4= plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')

ax4.plot_surface(x, y, z2(x,y),color='b',label='Iris Setosa')
ax4.plot_surface(x, y, z1(x,y),color='r',label='Iris Versicolour')
ax4.plot_surface(x, y, z3(x,y),color='g',label='Iris Virginica')
ax4.scatter(data_dims123[class0,0],data_dims123[class0,1],data_dims123[class0,2], c='r', marker='o')
ax4.scatter(data_dims123[class1,0],data_dims123[class1,1],data_dims123[class1,2], c='b', marker='o')
ax4.scatter(data_dims123[class2,0],data_dims123[class2,1],data_dims123[class2,2], c='g', marker='o')


ax4.set_title('Decision Boundaries Grouped')
ax4.set_xlabel('Sepal Width (m)')
ax4.set_ylabel('Petal Length (m)')
ax4.set_zlabel('Petal Width (m)')

plt.show()
