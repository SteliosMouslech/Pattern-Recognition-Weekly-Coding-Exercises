import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y=np.logical_xor(X[:,0],X[:,1])


#I tried for different C values
#we have polynomial of (x*y+coef0)^degree so for our example coef0=2
model = svm.SVC(kernel='poly',degree=2,coef0=0,C=3)
#model = svm.SVC(kernel='poly',degree=2,coef0=0,C=1)
#model = svm.SVC(kernel='poly',degree=2,coef0=0,C=2)
#model = svm.SVC(kernel='poly',degree=2,coef0=0,C=0.5)
clf = model.fit(X, Y)
fig, ax = plt.subplots()
# title for the plots
title ="XOR Problem with polynomial of degree 2 and coef=0 with C=3"
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=40, edgecolors="k")
ax.set_ylabel("x2")
ax.set_xlabel("x1")
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
plt.show()
