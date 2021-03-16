import numpy as np
import math



####################################  FOR EXERCISE 4.1###############################################

def parzen_gauss_kernel(x,h,low,high):
    dims=x.shape
    if len(dims)==1:
        l=1
        n=dims[0]
    else:
        l=dims[0]
        n=dims[1]

    if (high-low)/h>1:
        px=np.zeros(math.floor((high-low)/h))
    else:
        px=np.zeros(math.ceil((high-low)/h))

    k=0
    for i in np.arange(low,high,h):
        for z in range(n):
            if l==1:
                xi=x[z]
            else:
                xi=x[:,z]
            c=i-xi
            dotProd=np.dot(c.T,c)
            px[k]=px[k] + math.exp(-dotProd/(2*h**2))
        px[k]=px[k]*(1/n)*(1/(((2*math.pi)**(l/2))*(h**l)))
        k=k+1

    return px


def knn_density_estimate(x,knn,low,high,step):
    dims=x.shape
    if len(dims)==1:
        l=1
        n=dims[0]
    elif len(dims)>3:
        print("this function currently works up to 3d data :) ")
        return
    else:
        l=dims[0]
        n=dims[1]
    px=np.zeros(math.floor((high-low)/step))
    euclidianDistances=np.zeros(n)
    k=0
    for i in np.arange(low, high, step):
        for z in range(n):
            if l == 1:
                xi = x[z]
                euclidianDistances[z] = np.sqrt(np.sum((i - xi) ** 2))
            else:
                xi = x[:,z]
                euclidianDistances[z]=np.np.sqrt(np.sum((i-xi).T.dot(x-xi)))

        sortedDistances=np.sort(euclidianDistances)
        r=sortedDistances[knn-1] #indexing starts at 0 so first neighbor is [0]
        if l==1:  #1D
            v=2*r
        elif l==2: #2D
            v=2*math.pi*r**2
        else:      #3D
            v=(4/3)*math.pi*r**2
        px[k]=knn/(n*v)
        k=k+1

    return px


####################################  FOR EXERCISE 4.2###############################################

def createOnevariablelassData(numberOfClasses,datasize,p,listofm,listofs):
    listOfData=[]
    labellist=[]
    numberOfSamplesCreated=0 #check how many we have created so last class rounds to datasize!!!!
    for i in range(numberOfClasses-1):
        classSize=math.floor(datasize*p[i])
        listOfData.append(np.random.normal(listofm[i],listofs[i],size=classSize))
        numberOfSamplesCreated=numberOfSamplesCreated+classSize
        labellist.append(np.ones(classSize)*i)
    #for last class we need to take the remaining samples cause for sum of p we might not have round number of data
    lastclassSize=datasize-numberOfSamplesCreated
    listOfData.append(np.random.normal(listofm[-1],listofs[-1],size=lastclassSize))
    labellist.append(np.ones(lastclassSize) * (i+1))

    dataVector=np.concatenate(listOfData)
    labelVector=np.concatenate(labellist)


    return dataVector,labelVector


def k_nn_classifier(numberOfClasses,trainDataSet,trainDataSetLabels,knn,testDataset):
    dims_trainData=trainDataSet.shape
    dims_testData=testDataset.shape

    if len(dims_testData)!=len(dims_trainData):
        print("Train and test data variables must have the same number of dimensions! ")
        return
    else:
        if len(dims_trainData)==1: # we can also use testData dims because now we know they are equal
            l=1
            n_train=dims_trainData[0]
            n_test=dims_testData[0]
        else:
            l = dims_trainData[0]  #l is the same we can use train or test to get it
            n_train=dims_trainData[1]
            n_test =dims_testData[1]

    testDataLabels=np.zeros(n_test)

    for i in range(n_test):
        #euclidian squared here
        if len(dims_trainData)>1: #in casee our data are not 1D
            distances=np.sum(np.power(testDataset[:,i] -trainDataSet, 2),axis=0)
        else:#in case data is 1D we get absolut distance as euclidian distance
            distances=np.abs(testDataset[i] -trainDataSet)
        pointsNearest=np.argsort(distances)
        labelCount=np.zeros(numberOfClasses)
        for j in range (knn):
            classNo=int(trainDataSetLabels[pointsNearest[j]])
            labelCount[classNo]=labelCount[classNo]+1  ##we cound how many times each class is a neighbor (histogram like)

        testDataLabels[i]=np.argmax(labelCount)
    return testDataLabels



def errorRateCalculation(classPredicted,actualClass):
    if len(classPredicted)!= len(actualClass):
        print("the vectors with the  class labels are not same size  fix!!")
        return
    else:
        numberOfData=len(actualClass) #same with class predicted!
    wrong_classifications=0
    for i in range(numberOfData):
        if classPredicted[i] != actualClass[i]:
            wrong_classifications=wrong_classifications+1

    return wrong_classifications/numberOfData


def findBestKforTest(trainData,trainDataLabels,testData,testDataLabels,lowest_k,highest_k,maxTolerence):

    counter=0
    bestK=-1
    bestErrorRate=100
    for k in range(lowest_k,highest_k+1):
        knnLabels = k_nn_classifier(3, trainData, trainDataLabels, k, testData)
        errorRate = errorRateCalculation(knnLabels, testDataLabels)
        if errorRate < bestErrorRate:
            bestErrorRate=errorRate
            bestK=k
            counter=0
        elif counter>maxTolerence:
            return bestK,bestErrorRate
        else:
            counter=counter+1
    return bestK,bestErrorRate


    ##Taken from my Ex3_1

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
        g = -0.5 * c**2*invS- (d / 2) * math.log(2 * math.pi) - 0.5 * math.log(detS) + math.log(pw)
    return g


def bayesianClassifier(trainData,trainDataLabels,testData,apriori):
    numberOfClasses=len(apriori)
    listOfSplitData=[]
    #Calculate M and S for every class
    startingIndex=0
    listofmeans=[]
    listofS=[]
    for i in range (numberOfClasses):
        numberOfElementsInClass=np.count_nonzero(trainDataLabels==i)
        listofmeans.append(np.mean(trainData[startingIndex:startingIndex+numberOfElementsInClass]))
        listofS.append(np.cov(trainData[startingIndex:startingIndex+numberOfElementsInClass]))
        startingIndex=startingIndex+numberOfElementsInClass

    testDataLabels=np.zeros(len(testData))
    for j in range(len(testData)):
        biggest_g=-100000
        classOfDataPoint=-1
        for classNo in range(numberOfClasses):
            g=classifyFunc(1,testData[j],listofmeans[classNo],listofS[classNo],apriori[classNo])
            if g>biggest_g:
                biggest_g=g
                classOfDataPoint=classNo
        testDataLabels[j]=classOfDataPoint
    return testDataLabels



def parzenWindowClassifier(trainData,trainDataLabels,testData,h,apriori):

    numberOfClasses = len(apriori) # see how many classes we hve
    #split data of each train class
    startingIndex=0
    listofClassData=[]
    for i in range(numberOfClasses):
        numberOfElementsInClass = np.count_nonzero(trainDataLabels == i)
        dataofClass=trainData[startingIndex:startingIndex+numberOfElementsInClass]
        listofClassData.append(dataofClass)

        startingIndex=startingIndex+numberOfElementsInClass

    #now for each testData point we need to calculate likelihood P(wi/x) using Parzen Window
    #then we classify it using baysian rule P(wi)*P(wi/x) > P(wj)*P(wj/x)

    predicted_class=np.zeros(len(testData))
    for j in range(len(testData)):
        #we use our previous functtion to estimate this likelihood with a little trick to do it only at our test point
        classOfPoint =-1
        bestPosterior = -10000
        for classNo in range(numberOfClasses):
            likelihood_ofDataPoint = parzen_gauss_kernel(listofClassData[classNo],h,testData[j],testData[j]+(h/2)) #IMPORTANT! with low our data point and high our datapoint+h/2 it will only estimate it for our datapoint
            posterior=likelihood_ofDataPoint * apriori[classNo]
            if posterior>bestPosterior:
                classOfPoint=classNo
                bestPosterior=posterior
        predicted_class[j]=classOfPoint
    return predicted_class

def findBesthForParzen(trainData,trainDataLabels,testData,testDataLabels,p,lowest_h,highest_h,step,maxTolerence):
    counter=0
    best_h=-1
    bestErrorRate=100
    for h in np.arange(lowest_h,highest_h+step,step):
        parzenPredictions = parzenWindowClassifier(trainData, trainDataLabels, testData, h, p)
        parzenError = errorRateCalculation(parzenPredictions, testDataLabels)
        if parzenError < bestErrorRate:
            bestErrorRate=parzenError
            best_h=h
            counter=0
        elif counter>maxTolerence:
            return best_h,bestErrorRate
        else:
            counter=counter+1
    return best_h,bestErrorRate












