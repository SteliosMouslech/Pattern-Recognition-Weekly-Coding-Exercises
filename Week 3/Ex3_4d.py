import numpy as np
import math
import Ex3_1
np.random.seed(132)

def createMultivariateClassData(numberOfClasses,datasize,p,listofm,listofs):
    listOfData=[]
    numberOfSamplesCreated=0 #check how many we have created so last class rounds to datasize!!!!
    for i in range(numberOfClasses-1):
        classSize=math.floor(datasize*p[i])
        listOfData.append(np.random.multivariate_normal(listofm[i],listofs[i],size=classSize))
        numberOfSamplesCreated=numberOfSamplesCreated+classSize
    #for last class we need to take the remaining samples cause for sum of p we might not have round number of data
    lastclassSize=datasize-numberOfSamplesCreated
    listOfData.append(np.random.multivariate_normal(listofm[-1],listofs[-1],size=lastclassSize))
    return listOfData


trainsize=10000
testsize=1000

p1=1/6
p2=1/6
p3=2/3
listofP=[p1,p2,p3]

m1=np.array([0,0,0])
m2=np.array([1,2,2])
m3=np.array([3,3,4])
listofm=[m1,m2,m3]

s1=np.array([[0.8,0.2,0.1],[0.2,0.8,0.2],[0.1,0.2,0.8]])
s2=np.array([[0.6,0.2,0.01],[0.2,0.8,0.01],[0.01,0.01,0.6]])
s3=np.array([[0.6,0.1,0.1],[0.1,0.6,0.1],[0.1,0.1,0.6]])
listofS=[s1,s2,s3]


train_data=createMultivariateClassData(3,trainsize,listofP,listofm,listofS)
test_data=createMultivariateClassData(3,testsize,listofP,listofm,listofS)



#test our data now with m s from ekfonisi

#EUCLIDIAN DISTANCE

euclidianFalses=0
for i in range(test_data[0].shape[0]):
    #test of class1 -eucldian distance
    distanceto1=Ex3_1.euclidianDistance(test_data[0][i],m1,3)
    distanceto2=Ex3_1.euclidianDistance(test_data[0][i],m2,3)
    distanceto3=Ex3_1.euclidianDistance(test_data[0][i],m3,3)

    #we made mistake for class1 if distanceto1 is not the smallest
    if distanceto1>distanceto2 or distanceto1>distanceto3:
        euclidianFalses= euclidianFalses + 1

#do the same for the rest of classes
for i in range(test_data[1].shape[0]):
    # test of class2 -euclidian distance
    distanceto1 = Ex3_1.euclidianDistance(test_data[1][i], m1,3)
    distanceto2 = Ex3_1.euclidianDistance(test_data[1][i], m2,3)
    distanceto3 = Ex3_1.euclidianDistance(test_data[1][i], m3, 3)

    # we made mistake for class2 if distanceto2 is not the smallest
    if distanceto2 > distanceto1 or distanceto2 > distanceto3:
        euclidianFalses = euclidianFalses + 1


for i in range(test_data[2].shape[0]):
    # test of class3 -euclidian distance
    distanceto1 = Ex3_1.euclidianDistance(test_data[2][i], m1, 3)
    distanceto2 = Ex3_1.euclidianDistance(test_data[2][i], m2, 3)
    distanceto3 = Ex3_1.euclidianDistance(test_data[2][i], m3, 3)

    # we made mistake for class2 if distanceto3 is not the smallest
    if distanceto3 > distanceto1 or distanceto3 > distanceto2:
        euclidianFalses = euclidianFalses + 1

euclidianError=euclidianFalses/testsize

print("Euclidian Minimun Distance Classifier Error:",euclidianError," Using theoretical Mean and Covariance")


#MAHALANOBIS CLASSIFIER

mahalanobisFalses=0
for i in range(test_data[0].shape[0]):
    #test of class1 -mahalanobis distance
    distanceto1=Ex3_1.mahalanobisDistance(test_data[0][i],m1,s1,3)
    distanceto2=Ex3_1.mahalanobisDistance(test_data[0][i],m2,s2,3)
    distanceto3=Ex3_1.mahalanobisDistance(test_data[0][i],m3,s3,3)

    #we made mistake for class1 if distanceto1 is not the smallest
    if distanceto1>distanceto2 or distanceto1>distanceto3:
        mahalanobisFalses=mahalanobisFalses+1
    #do the same for the rest of classes

for i in range(test_data[1].shape[0]):
    # test of class2 -mahalanobis distance
    distanceto1 = Ex3_1.mahalanobisDistance(test_data[1][i], m1, s1, 3)
    distanceto2 = Ex3_1.mahalanobisDistance(test_data[1][i], m2, s2, 3)
    distanceto3 = Ex3_1.mahalanobisDistance(test_data[1][i], m3, s3, 3)

    # we made mistake for class2 if distanceto2 is not the smallest
    if distanceto2 > distanceto1 or distanceto2 > distanceto3:
        mahalanobisFalses = mahalanobisFalses + 1


for i in range(test_data[2].shape[0]):
    # test of class3 -mahalanobis distance
    distanceto1 = Ex3_1.mahalanobisDistance(test_data[2][i], m1, s1, 3)
    distanceto2 = Ex3_1.mahalanobisDistance(test_data[2][i], m2, s2, 3)
    distanceto3 = Ex3_1.mahalanobisDistance(test_data[2][i], m3, s3, 3)

    # we made mistake for class2 if distanceto3 is not the smallest
    if distanceto3 > distanceto1 or distanceto3 > distanceto2:
        mahalanobisFalses = mahalanobisFalses + 1

mahalanobisError=mahalanobisFalses/testsize

print("Mahalanobis Minimun Distance Classifier Error:",mahalanobisError," Using theoretical Mean and Covariance")


#Bayesian Classifier
# test of class1 - bayesian classifier
bayesianFalses=0
for i in range(test_data[0].shape[0]):
    d1=Ex3_1.classifyFunc(3,test_data[0][i],m1,s1,p1)
    d2=Ex3_1.classifyFunc(3,test_data[0][i],m2,s2,p2)
    d3=Ex3_1.classifyFunc(3,test_data[0][i],m3,s3,p3)

    #we made mistake for class1 if d1 not the Biggest
    if d1<d2 or d1<d3:
        bayesianFalses=bayesianFalses+1
#do the same for the rest of classes
for i in range(test_data[1].shape[0]):
    d1=Ex3_1.classifyFunc(3,test_data[1][i],m1,s1,p1)
    d2=Ex3_1.classifyFunc(3,test_data[1][i],m2,s2,p2)
    d3=Ex3_1.classifyFunc(3,test_data[1][i],m3,s3,p3)

    #we made mistake for class1 if d2 not the Biggest
    if d2<d1 or d2<d3:
        bayesianFalses=bayesianFalses+1

#do the same for the rest of classes
for i in range(test_data[2].shape[0]):
    d1=Ex3_1.classifyFunc(3,test_data[2][i],m1,s1,p1)
    d2=Ex3_1.classifyFunc(3,test_data[2][i],m2,s2,p2)
    d3=Ex3_1.classifyFunc(3,test_data[2][i],m3,s3,p3)

    #we made mistake for class1 if d3 not the Biggest
    if d3<d1 or d3<d2:
        bayesianFalses=bayesianFalses+1

bayesianError=bayesianFalses/testsize
print("Bayesian Classifier Error:",bayesianError," Using theoretical Mean and Covariance")








#G EROTIMA
print("\nEROTIMA G\n")

#calculate means and covariance matrices from our training data
class1mean=np.mean(train_data[0],0)
class2mean=np.mean(train_data[1],0)
class3mean=np.mean(train_data[2],0)
class1cov=np.cov(train_data[0],rowvar=False) #rowvar=False means that rows are observations and cols are variables
class2cov=np.cov(train_data[1],rowvar=False)
class3cov=np.cov(train_data[2],rowvar=False)


euclidianFalses=0
for i in range(test_data[0].shape[0]):
    #test of class1 -eucldian distance
    distanceto1=Ex3_1.euclidianDistance(test_data[0][i],class1mean,3)
    distanceto2=Ex3_1.euclidianDistance(test_data[0][i],class2mean,3)
    distanceto3=Ex3_1.euclidianDistance(test_data[0][i],class3mean,3)

    #we made mistake for class1 if distanceto1 is not the smallest
    if distanceto1>distanceto2 or distanceto1>distanceto3:
        euclidianFalses= euclidianFalses + 1

#do the same for the rest of classes
for i in range(test_data[1].shape[0]):
    # test of class2 -euclidian distance
    distanceto1 = Ex3_1.euclidianDistance(test_data[1][i], class1mean,3)
    distanceto2 = Ex3_1.euclidianDistance(test_data[1][i], class2mean,3)
    distanceto3 = Ex3_1.euclidianDistance(test_data[1][i], class3mean, 3)

    # we made mistake for class2 if distanceto2 is not the smallest
    if distanceto2 > distanceto1 or distanceto2 > distanceto3:
        euclidianFalses = euclidianFalses + 1


for i in range(test_data[2].shape[0]):
    # test of class3 -euclidian distance
    distanceto1 = Ex3_1.euclidianDistance(test_data[2][i], class1mean, 3)
    distanceto2 = Ex3_1.euclidianDistance(test_data[2][i], class2mean, 3)
    distanceto3 = Ex3_1.euclidianDistance(test_data[2][i], class3mean, 3)

    # we made mistake for class2 if distanceto3 is not the smallest
    if distanceto3 > distanceto1 or distanceto3 > distanceto2:
        euclidianFalses = euclidianFalses + 1

euclidianError=euclidianFalses/testsize

print("Euclidian Minimun Distance Classifier Error:",euclidianError," Using calculated Maximun Likelihood Mean and Covariance")










mahalanobisFalses=0
# test of class1 -mahalanobis distance
for i in range(test_data[0].shape[0]):
    #test of class1 -mahalanobis distance
    distanceto1=Ex3_1.mahalanobisDistance(test_data[0][i],class1mean,class1cov,3)
    distanceto2=Ex3_1.mahalanobisDistance(test_data[0][i],class2mean,class2cov,3)
    distanceto3=Ex3_1.mahalanobisDistance(test_data[0][i],class3mean,class3cov,3)
    #we made mistake for class1 if distanceto1 is not the smallest
    if distanceto1>distanceto2 or distanceto1>distanceto3:
        mahalanobisFalses=mahalanobisFalses+1
 #do the same for the rest of classes
for i in range(test_data[1].shape[0]):
    # test of class2 -mahalanobis distance
    distanceto1 = Ex3_1.mahalanobisDistance(test_data[1][i],class1mean,class1cov,3)
    distanceto2 = Ex3_1.mahalanobisDistance(test_data[1][i],class2mean,class2cov,3)
    distanceto3 = Ex3_1.mahalanobisDistance(test_data[1][i],class3mean,class3cov,3)
    # we made mistake for class2 if distanceto2 is not the smallest
    if distanceto2 > distanceto1 or distanceto2 > distanceto3:
        mahalanobisFalses = mahalanobisFalses + 1
for i in range(test_data[2].shape[0]):
    # test of class3 -mahalanobis distance
    distanceto1 = Ex3_1.mahalanobisDistance(test_data[2][i],class1mean,class1cov,3)
    distanceto2 = Ex3_1.mahalanobisDistance(test_data[2][i],class2mean,class2cov,3)
    distanceto3 = Ex3_1.mahalanobisDistance(test_data[2][i],class3mean,class3cov,3)
    # we made mistake for class2 if distanceto3 is not the smallest
    if distanceto3 > distanceto1 or distanceto3 > distanceto2:
        mahalanobisFalses = mahalanobisFalses + 1

mahalanobisError=mahalanobisFalses/testsize
print("Mahalanobis Minimun Distance Classifier Error:",mahalanobisError," Using calculated Maximun Likelihood Mean and Covariance")



#Bayesian Classifier
# test of class1 - bayesian classifier
bayesianFalses=0
for i in range(test_data[0].shape[0]):
    d1=Ex3_1.classifyFunc(3,test_data[0][i],class1mean,class1cov,p1)
    d2=Ex3_1.classifyFunc(3,test_data[0][i],class2mean,class2cov,p2)
    d3=Ex3_1.classifyFunc(3,test_data[0][i],class3mean,class3cov,p3)

    #we made mistake for class1 if d1 not the Biggest
    if d1<d2 or d1<d3:
        bayesianFalses=bayesianFalses+1
#do the same for the rest of classes
for i in range(test_data[1].shape[0]):
    d1=Ex3_1.classifyFunc(3,test_data[1][i],class1mean,class1cov,p1)
    d2=Ex3_1.classifyFunc(3,test_data[1][i],class2mean,class2cov,p2)
    d3=Ex3_1.classifyFunc(3,test_data[1][i],class3mean,class3cov,p3)

    #we made mistake for class1 if d2 not the Biggest
    if d2<d1 or d2<d3:
        bayesianFalses=bayesianFalses+1

#do the same for the rest of classes
for i in range(test_data[2].shape[0]):
    d1=Ex3_1.classifyFunc(3,test_data[2][i],class1mean,class1cov,p1)
    d2=Ex3_1.classifyFunc(3,test_data[2][i],class2mean,class2cov,p2)
    d3=Ex3_1.classifyFunc(3,test_data[2][i],class3mean,class3cov,p3)
    #we made mistake for class1 if d3 not the Biggest
    if d3<d1 or d3<d2:
        bayesianFalses=bayesianFalses+1

bayesianError=bayesianFalses/testsize
print("Bayesian Classifier Error:",bayesianError," Using calculated Maximun Likelihood Mean and Covariance")