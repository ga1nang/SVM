import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC


#load dataset
mat = loadmat('data\\product_c_.mat')
X = mat["X"]
y = mat["y"]


#plot the dataset
m,n = X.shape[0],X.shape[1]
pos2,neg2= (y==1).reshape(m,1), (y==0).reshape(m,1)
plt.figure(figsize=(8,6))
plt.scatter(X[pos2[:,0],0],X[pos2[:,0],1],c="r",marker="+")
plt.scatter(X[neg2[:,0],0],X[neg2[:,0],1],c="y",marker="o")
plt.xlim(0,1)
plt.ylim(0.4,1)
plt.show()


#SVC model
classifier = SVC(C=100, kernel='rbf', gamma=300)
classifier.fit(X, y.ravel())


#plot the graph with SVC
plt.figure(figsize=(8,6))
plt.scatter(X[pos2[:,0],0],X[pos2[:,0],1],c="r",marker="+")
plt.scatter(X[neg2[:,0],0],X[neg2[:,0],1],c="y",marker="o")
X_5,X_6 = np.meshgrid(np.linspace(X[:,0].min(),X[:,1].max(),num=100),np.linspace(X[:,1].min(),X[:,1].max(),num=100))
plt.contour(X_5,X_6,classifier.predict(np.array([X_5.ravel(),X_6.ravel()]).T).reshape(X_5.shape),1,colors="b")
plt.xlim(0,1)
plt.ylim(0.4,1)
plt.show()