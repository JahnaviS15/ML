# from numpy import *
from os import listdir
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np1
import numpy.linalg as np
from scipy.stats import pearsonr
 
def kernel(point,xmat, k):
    m,n = np1.shape(xmat)
    weights = np1.asmatrix(np1.eye((m))) # eye(m) returns identity matrix of size m
    for j in range(m):
        diff = point - X[j]
        weights[j,j] = np1.exp(diff*diff.T/(-2.0*k**2))
    return weights
 
def localWeight(point,xmat,ymat,k):
    wei = kernel(point,xmat,k)
    XT_WX = X.T * (wei * X)
    XT_WX += np1.eye(XT_WX.shape[0]) * 1e-5  # Regularization
    W = XT_WX.I * (X.T * (wei * ymat.T))
    return W
     
def localWeightRegression(xmat,ymat,k):
    m,n = np1.shape(xmat)
    ypred = np1.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i]*localWeight(xmat[i],xmat,ymat,k)
    return ypred
       
# load data points
data = pd.read_csv('tips.csv')
bill = np1.array(data.total_bill)
tip = np1.array(data.tip)
 
#preparing and add 1 in bill
mbill = np1.asmatrix(bill) # convert array into matrix
mtip = np1.asmatrix(tip) # mat is used to convert to n dimesiona to 2 dimensional array form
m= np1.shape(mbill)[1]

# print(m) 244 data is stored in m
one = np1.asmatrix(np1.ones(m)) 
X= np1.hstack((one.T,mbill.T)) # create a stack of bill from ONE
# print(X)
#set k here

ypred = localWeightRegression(X,mtip,0.3) 
SortIndex = X[:,1].argsort(0)
xsort = X[SortIndex][:,0]
     
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(1,1,1)
ax.scatter(bill,tip, color='blue', label = 'Actual Data')
ax.plot(xsort[:,1],ypred[SortIndex], color = 'red', linewidth=2, label='Predicted Regression Line')
plt.xlabel('Total bill', fontsize = 14)
plt.ylabel('Tip', fontsize = 14)

plt.title('Locally Weighted Regression', fontsize=18)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('LWRReg.png')
plt.show()