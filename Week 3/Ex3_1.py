import numpy as np
import math

def classifyFunc(d,x,m,s,pw):
    c = x - m  # not necesseary but to have cleaner code in g

    #np.det() and np.linalg.inv()  input must be at least 2D array so we must check first for dimension of our distribution
    if d>1: #that means we have at least 2D so S is a 2D matrix
        detS=round(abs(np.linalg.det(s)),4) #roundind is needed cause linalg.det sometimes returns floating points with 1 bit rounding error
        invS=np.linalg.inv(s)
        g = -0.5 * (c.T).dot(invS).dot(c) - (d / 2) * math.log(2 * math.pi) - 0.5 * math.log(detS) + math.log(pw)
    else:
        detS=abs(s)
        invS=s**-1
        g = -0.5 * c**2*invS- (d / 2) - (d / 2) * math.log(2 * math.pi) - 0.5 * math.log(detS) + math.log(pw)
    return g

def euclidianDistance(x1,x2,d):
    if d==1:
        distance= np.sqrt(np.sum(np.square(x1 - x2)))
    else:
        distance=np.sqrt(np.sum((x1-x2).T.dot(x1-x2)))
    return distance

def mahalanobisDistance(x,m,s,d):
    c=x-m
    if d>1:
        invS=np.linalg.inv(s)
        distance = np.sqrt(c.T.dot(invS).dot(c))
    else:
        invS = s ** -1
        distance = math.sqrt(c*invS*c)
    return distance