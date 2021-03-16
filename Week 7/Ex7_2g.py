import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.neural_network import MLPClassifier # neural network

from sklearn.datasets import load_iris

iris=load_iris()
X=iris.data
Y=iris.target

print(25*" ","Multi Layer Perceptron - Two hidden layer")
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0,stratify=Y)

for number_of_neurons in [2,5,10,50]:
    print(50*"==")
    print("Number of Neurons in Hidden Layer=", number_of_neurons,"\nActivation Function:logistic")

    model=MLPClassifier(hidden_layer_sizes=(number_of_neurons,number_of_neurons,),solver='sgd',activation='logistic',learning_rate_init=0.01,max_iter=10000,random_state=0,tol=1e-4,)
    model.fit(X_train,y_train)
    predict_train = model.predict(X_train)
    print("Train accuracy:", accuracy_score(predict_train, y_train))
    scores = cross_val_score(model, X_train, y_train, cv=4)
    print("Validation Accuracy ", ": %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    predict=model.predict(X_test)
    print("Test accuracy:",accuracy_score(y_test,predict))


for number_of_neurons in [2,5,10,50]:
    print(50*"==")
    print("Number of Neurons in Each Hidden Layer=", number_of_neurons,"\nActivation Function:relu")

    model=MLPClassifier(hidden_layer_sizes=(number_of_neurons,number_of_neurons,),solver='sgd',activation='relu',learning_rate_init=0.01,max_iter=10000,random_state=0,tol=1e-4)
    model.fit(X_train,y_train)
    predict_train = model.predict(X_train)
    print("Train accuracy:", accuracy_score(predict_train, y_train))
    scores = cross_val_score(model, X_train, y_train, cv=4)
    print("Validation Accuracy ", ": %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    predict=model.predict(X_test)
    print("Test accuracy:",accuracy_score(y_test,predict))



for number_of_neurons in [2,5,10,50]:
    print(50*"==")
    print("Number of Neurons in Each Hidden Layer=", number_of_neurons,"\nActivation Function:tanh")

    model=MLPClassifier(hidden_layer_sizes=(number_of_neurons,number_of_neurons,),solver='sgd',activation='tanh',learning_rate_init=0.001,max_iter=10000,random_state=0,tol=1e-4)
    model.fit(X_train,y_train)
    predict_train = model.predict(X_train)
    print("Train accuracy:", accuracy_score(predict_train, y_train))
    scores = cross_val_score(model, X_train, y_train, cv=4)
    print("Validation Accuracy ", ": %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    predict=model.predict(X_test)
    print("Test accuracy:",accuracy_score(y_test,predict))


