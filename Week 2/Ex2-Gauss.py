import numpy as np

np.random.seed(1312)

def gaussDecision(numberOfSamples):
    p1=1/3

    #sample size for our samples according to our  proppabilities
    w1_sample_size=int(numberOfSamples*p1)
    w2_sample_size=numberOfSamples-w1_sample_size
    lamdas=[1,2,3,1]
    #create our samples according to mean and standard deviation we have from our data

    w1=np.random.normal(2,np.sqrt(0.5),w1_sample_size)
    w2=np.random.normal(1.5,np.sqrt(0.2),w2_sample_size)
    w1ClassifiedAsw2=0
    w1ClassifiedAsw1 = 0
    w2ClassifiedAsw1=0
    w2ClassifiedAsw2 = 0
    #for actual w1 class data we classify according to our decision boundaries at x=0.403, x=19303
    for point in w1:
        if point<1.9303 and point>0.403:
            w1ClassifiedAsw2+=1
        else:
            w1ClassifiedAsw1+=1

        # for actual w1 class data we classify according to our decision boundaries at x=0.403, x=19303
    for point2 in w2:
        if point2 < 1.9303 and point2 > 0.403:
            w2ClassifiedAsw2 +=1
        else:
            w2ClassifiedAsw1 +=1

    #calculate the propabbilities
    p_w1ClassifiedAsw1= w1ClassifiedAsw1 / w1_sample_size
    p_w1ClassifiedAsw2 = w1ClassifiedAsw2 /w1_sample_size
    p_w2ClassifiedAsw1 = w2ClassifiedAsw1 / w2_sample_size
    p_w2ClassifiedAsw2 = w2ClassifiedAsw2 / w2_sample_size

    print("Percentage of w1 classified as w1:",p_w1ClassifiedAsw1)
    print("Percentage of w1 classified as w2:",p_w1ClassifiedAsw2)
    print("Percentage of w2 classified as w1:", p_w2ClassifiedAsw1)
    print("Percentage of w2 classified as w2", p_w2ClassifiedAsw2)
    #the cost according to theory
    cost=(((lamdas[0]*p_w1ClassifiedAsw1)+(lamdas[1]*p_w1ClassifiedAsw2))*p1)+ (((lamdas[2]*p_w2ClassifiedAsw1)+(lamdas[3]*p_w2ClassifiedAsw2))*(1-p1))
    print("The cost is:",cost)


gaussDecision(15000)