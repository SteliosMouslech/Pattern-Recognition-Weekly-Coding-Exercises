import numpy as np
import math

def duda_batch_Perceptron(data, dataLabels, learningRate, epochs, init_weights):
    if init_weights.shape[0] != data.shape[1]:
        print("check dims of w or data")
        return

    erros_per_epoch = []
    N,l  = data.shape
    data = np.hstack((data, np.ones((data.shape[0],1), dtype=data.dtype)))
    weights = np.concatenate([init_weights,[1.0]])
    iters = 0
    errors = 1
    while iters in range(epochs) and errors > 0:
        errors = 0

        gradi = np.zeros(l+1)
        for i in range(N):
            x = data[i,:]  # this makes X an 2d row!!!
            if (x.dot(weights))*dataLabels[i] < 0:
                errors += 1
                gradi=gradi+ learningRate*(-dataLabels[i]*x)

        weights=weights - learningRate*gradi
        erros_per_epoch.append(errors)
        iters+=1

    return weights,erros_per_epoch,iters

def duda_batch_Perceptron_with_Margin(data, dataLabels,b, learningRate, epochs):
    c,n= data.shape
    data = np.vstack((data, np.ones((1,data.shape[1]), dtype=data.dtype)))
    train_zero=np.argwhere(dataLabels==0)
    #augment my daata
    augmentData=np.copy(data)
    for i in train_zero:
        augmentData[:,i]=-augmentData[:,i]

    weights = np.sum(augmentData, axis=1)/n

    yk=[0]
    iter=0
    errors_perepoch=[]
    while yk and iter<epochs:
        iter += 1
        yk=[]
        errors=0

        for k in range(n):
            x=augmentData[:,k]
            if weights.dot(x) <= b:
                yk.append(k)
                errors=errors+1
        errors_perepoch.append(errors)
        if not yk:
            break

        y=augmentData[:,yk]
        var1 = b - weights.dot(y)
        var2= np.sum(y**2,axis=0)
        grad=var1/var2
        grad=np.atleast_2d(grad) # so matrix mul works correctly..
        test=np.ones((c+1,1)).dot(grad)
        test2=test*y
        update=np.sum(test2,axis=1)
        weights=weights + learningRate*update



    return weights,iter,errors_perepoch



def  duda_LMS(data,dataLabels,max_iter,theta,learningRate):
    c, n = data.shape
    data = np.vstack((data, np.ones((1, data.shape[1]), dtype=data.dtype)))


    # augment my daata
    augmentData = np.copy(data)
    train_zero=np.argwhere(dataLabels==0)
    b = 2*dataLabels -1
    for i in train_zero:
        augmentData[:, i] = -augmentData[:, i]
    weights=np.sum(augmentData,axis=1)/n
    update=100
    updates=[]
    k=0
    iter=0

    while np.sum(np.abs(update)) > theta and iter<max_iter :
        iter +=1
        #changing a bit cause python arrays index at 0
        y=data[:,k]
        var=learningRate*(b[k] - weights.dot(y))
        update = var*y
        weights=weights+update
        updates.append(np.sum(np.abs(update)))
        k+=1
        if k==n:
            k=0

    return weights,updates



def duda_Ho_Kashyap(data,labels,type,max_iter,b_min,learningRate):

    c,n=data.shape
    train_class2=np.squeeze(np.argwhere(labels==-1))
    Y=data
    Y[:,train_class2]=-Y[:,train_class2]
    b=np.ones((1,n))
    weights=(np.linalg.pinv(Y.T)).dot(b.T)
    k=0
    e=1000
    found=0
    test=0
    criterion=np.sum(np.greater(np.abs(e),b_min))
    while criterion>0 and (k<max_iter) and (not found):
        k=k+1
        var=(Y.T).dot(weights)
        e=var.T-b
        e_plus=0.5*(e+np.abs(e))
        b=b+ (2*learningRate*e_plus)

        if type==0:
            weights=(np.linalg.pinv(Y.T)).dot(b.T)

        else:
            weights=weights+ learningRate*((np.linalg.pinv(Y.T)).dot(e_plus.T))
        criterion=np.sum(np.greater(np.abs(e),b_min))



    if k == max_iter:
        print("Algorithm did not find solution")
    else:
        print("found solution after", k, " iterations")
    return weights,b




def mperceptron(data,labels,max_iter):

    dim,num_data=data.shape
    nclass=np.max(labels)+1
    W=np.zeros((dim,nclass))
    b=np.zeros((nclass,1))
    t=0
    flag=0
    while max_iter>t and flag==0:
        t+=1
        flag=1
        for i in range(nclass):
            class_i=np.argwhere(labels==i)
            class_i=np.squeeze(class_i)
            dfce_i= np.dot(W[:,i].T,data[:,class_i])+b[i,0]

            for j in np.setdiff1d(np.arange(start=0,stop=nclass),np.atleast_1d(i)):
                dfce_j=np.dot(W[:,j].T,data[:,class_i])+b[j,0]
                min_dif=np.min(dfce_i-dfce_j,axis=0)
                inx=np.argmin(dfce_i-dfce_j,axis=0)
                inx=np.atleast_1d(inx)
                if min_dif  <=0:

                    inx=class_i[inx[0]]
                    W[:,i]=W[:,i]+ data[:,inx]
                    b[i,0]=b[i,0]+1

                    W[:, j] = W[:, j] - data[:, inx]
                    b[j,0] = b[j,0] - 1
                    flag=0
                    break
            if flag==0:
                break

    return W,b,t




