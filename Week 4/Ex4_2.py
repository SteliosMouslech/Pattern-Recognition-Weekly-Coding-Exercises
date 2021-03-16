import numpy as np
import Ex4_myFunctions as ex4
import matplotlib.pyplot as plt
import math




np.random.seed(21)





listofm=[2,1,3]
listofstdDeviation=[math.sqrt(0.5),math.sqrt(0.1),math.sqrt(1.3)]

p=[0.5,0.3,0.2]

trainDataSize=100
testDataSize=1000

trainData,trainDataLabels=ex4.createOnevariablelassData(3,trainDataSize,p,listofm,listofstdDeviation)
testData,testDataLabels=ex4.createOnevariablelassData(3,testDataSize,p,listofm,listofstdDeviation)





print("====KNN CLASSIFIER====")
for k in range(1,4):
    knnLabels=ex4.k_nn_classifier(3,trainData,trainDataLabels,k,testData)
    errorRate=ex4.errorRateCalculation(knnLabels,testDataLabels)
    print("K=",k, "KNN Error Percentage: ",errorRate)

print("Trying to find the best K...")

print("Trying a rule of thumb of k=sqrt(numberofTrainData)=10")
knnLabels = ex4.k_nn_classifier(3, trainData, trainDataLabels, 10, testData)
errorRate = ex4.errorRateCalculation(knnLabels, testDataLabels)
print("Error Rate for K=10 is: ",errorRate)



print("Using my 'brute force' method...")
k_best,error_best=ex4.findBestKforTest(trainData,trainDataLabels,testData,testDataLabels,1,100,20)

print("Best K is: ",k_best," With Error Percentage: ",error_best)


print("====BAYESIAN CLASSIFIER====")

bayesianPredicted=ex4.bayesianClassifier(trainData,trainDataLabels,testData,p)
bayesianError=ex4.errorRateCalculation(bayesianPredicted,testDataLabels)
print("Bayesian Classifier Error Rate  is: ",bayesianError)



print("====PARZEN WINDOW CLASSIFIER====")
for h in [0.1,0.3,0.5,0.7]:
    parzenPredictions=ex4.parzenWindowClassifier(trainData,trainDataLabels,testData,h,p)
    parzenError=ex4.errorRateCalculation(parzenPredictions,testDataLabels)
    print("h=",h, "Parzen Error Percentage: ",parzenError)


print("Using my 'brute force' method...")
h_best,error_best=ex4.findBesthForParzen(trainData,trainDataLabels,testData,testDataLabels,p,0.1,7,0.1,10)
print("Best h is: ",h_best," With Error Percentage: ",error_best)



##for PNN
##Ιgnore this code its only because i have numpy 1.13 for another reason so i want to ignore future warnings from tensorflow
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from neupy import algorithms as algo

print("====PNN CLASSIFIER====")

for hspread in [0.1,0.3,0.5,0.7]:
    pnn = algo.PNN(std=hspread, verbose=False)
    pnn.train(trainData,trainDataLabels)
    pnnPredicted=pnn.predict(testData)
    pnnError = ex4.errorRateCalculation(pnnPredicted, testDataLabels)
    print("hspread=",hspread, "PNN Error Percentage: ",pnnError)


