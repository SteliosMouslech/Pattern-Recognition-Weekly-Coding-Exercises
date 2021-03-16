import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


from sklearn.datasets import load_iris

iris=load_iris()
X=iris.data
Y=iris.target


X=X[:,[0,1,3]]

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0,stratify=Y)



print(25*" ","USING LINEAR SVM")
print(50 * "==")
for c in [0.5,1,10,100,1000]:
    model = svm.SVC(kernel='linear', C=c,decision_function_shape="ovr").fit(X_train, y_train,)
    scores = cross_val_score(model,X_train, y_train, cv=4)
    print("USING C=",c)
    print("Validation Accuracy ",": %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    predict=model.predict(X_test)
    print("Test accuracy:",accuracy_score(y_test,predict))
    print(50*"==")


print(25*" ","USING NON LINEAR SVM WITH RBF KERNEL")
print(50 * "==")
for c in [0.5, 1,10, 100, 1000]:
    model = svm.SVC(kernel='rbf', C=c,decision_function_shape="ovr").fit(X_train, y_train, )
    scores = cross_val_score(model, X_train, y_train, cv=4)
    print("USING C=",c)
    print("Validation Accuracy ", ": %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    predict = model.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, predict))
    print(50 * "==")



print(25*" ","USING NON LINEAR SVM WITH SIGMOID KERNEL")
print(50 * "==")
for c in [0.5, 1,10, 100, 1000]:
    model = svm.SVC(kernel='sigmoid', C=c,decision_function_shape="ovr").fit(X_train, y_train, )
    scores = cross_val_score(model, X_train, y_train, cv=4)
    print("USING C=",c)
    print("Validation Accuracy ", ": %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    predict = model.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, predict))
    print(50 * "==")



print(25*" ","USING NON LINEAR SVM WITH POLY KERNEL")
print(50 * "==")
for d in [2,4,6]:

    for c in [0.5, 1, 10, 100, 1000]:
        print("USING DEGREE=",d,"C=", c)
        model = svm.SVC(kernel='poly', C=c,decision_function_shape="ovr").fit(X_train, y_train, )
        scores = cross_val_score(model, X_train, y_train, cv=4)
        print("Validation Accuracy ", ": %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        predict = model.predict(X_test)
        print("Test accuracy:", accuracy_score(y_test, predict))
        print(50 * "==")