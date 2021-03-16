import numpy as np
import math
import Ex3_1

#get our data from txt file for ease of use
my_data = np.genfromtxt('data.txt', delimiter='')

#split our data to each class
classW1=np.zeros((10,3))
classW2=np.zeros((10,3))
classW3=np.zeros((10,3))

for i in range(10):
    classW1[i,0:3]=my_data[i,1:4]
    classW2[i, 0:3] = my_data[i, 4:7]
    classW3[i, 0:3] = my_data[i, 7:]

#we need to calculate mean,and covariance values first for only 1D (x1 only)

onedimMeanW1=np.mean(classW1[:,0])
onedimMeanW2=np.mean(classW2[:,0])
onedimCovarianceW1=np.cov(classW1[:,0],rowvar=False)
onedimCovarianceW2=np.cov(classW2[:,0],rowvar=False)


for n in np.arange(-9,9, 0.0001):
    y=Ex3_1.classifyFunc(1, n, onedimMeanW1, onedimCovarianceW1, 0.5) - Ex3_1.classifyFunc(1, n, onedimMeanW2, onedimCovarianceW2,0.5)
    if abs(y)<0.00001:
        print("G1(x)-G2(x)=0 for x: ", n,"","with error tolerance: ",y)

wrong=0
for i in range(10):
    if Ex3_1.classifyFunc(1,classW1[i,0],onedimMeanW1,onedimCovarianceW1, 0.5)<Ex3_1.classifyFunc(1,classW1[i,0],onedimMeanW2,onedimCovarianceW2,0.5):
        wrong=wrong+1

    if Ex3_1.classifyFunc(1,classW2[i,0],onedimMeanW1,onedimCovarianceW1, 0.5)>Ex3_1.classifyFunc(1,classW2[i,0],onedimMeanW2,onedimCovarianceW2,0.5):
        wrong=wrong+1
errorUsingx1=wrong/20
print("Classification error using only x1: ",errorUsingx1)


#lets use 2 characteristics now

twodimMeanW1=np.mean(classW1[:,0:2],axis=0)
twodimMeanW2=np.mean(classW2[:,0:2],axis=0)
twodimCovarianceW1=np.cov(classW1[:,0:2],rowvar=False)
twodimCovarianceW2=np.cov(classW2[:,0:2],rowvar=False)

wrong=0
for i in range(10):
    if Ex3_1.classifyFunc(2,classW1[i,0:2],twodimMeanW1,twodimCovarianceW1, 0.5)<Ex3_1.classifyFunc(2,classW1[i,0:2],twodimMeanW2,twodimCovarianceW2,0.5):
        wrong=wrong+1

    if Ex3_1.classifyFunc(2,classW2[i,0:2],twodimMeanW1,twodimCovarianceW1, 0.5)>Ex3_1.classifyFunc(2,classW2[i,0:2],twodimMeanW2,twodimCovarianceW2,0.5):
        wrong=wrong+1
errorUsingx1x2=wrong/20
print("Classification error using only x1,x2: ",errorUsingx1x2)


#lets use 3 characteristics now

threedimMeanW1=np.mean(classW1[:,0:3],axis=0)
threedimMeanW2=np.mean(classW2[:,0:3],axis=0)


threedimCovarianceW1=np.cov(classW1[:,0:3],rowvar=False)
threedimCovarianceW2=np.cov(classW2[:,0:3],rowvar=False)




wrong=0
for i in range(10):
    if Ex3_1.classifyFunc(3,classW1[i,0:3],threedimMeanW1,threedimCovarianceW1, 0.5)<Ex3_1.classifyFunc(3,classW1[i,0:3],threedimMeanW2,threedimCovarianceW2,0.5):
        wrong=wrong+1

    if Ex3_1.classifyFunc(3,classW2[i,0:3],threedimMeanW1,threedimCovarianceW1, 0.5)>Ex3_1.classifyFunc(3,classW2[i,0:3],threedimMeanW2,threedimCovarianceW2,0.5):
        wrong=wrong+1
errorUsingx1x2x3=wrong/20
print("Classification error using  x1,x2,x3: ",errorUsingx1x2x3)


#to calculate g1,g2,g3 symbolically we need values of m1,m2,m3,s1,s2,s3

threedimMeanW3=np.mean(classW3[:,0:3],axis=0)
threedimCovarianceW3=np.cov(classW3[:,0:3],rowvar=False)
print(threedimMeanW1)
print("")
print(threedimMeanW2)
print("")
print(threedimMeanW3)
print("")
print(threedimCovarianceW1)
print("")
print(threedimCovarianceW2)
print("")
print(threedimCovarianceW3)
print("")