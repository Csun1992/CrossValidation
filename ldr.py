import sys
import numpy as np
import scipy as sp
import math 
from random import shuffle

def sTildeInv(covar, dataDim, sampleSize):
    firstPart = (1-tUEye(covar, dataDim, sampleSize)) * (sampleSize-dataDim-2) * np.linalg.inv(covar)
    secondPart = tUEye(covar, dataDim, sampleSize) * (dataDim*sampleSize-dataDim-2) / \
        np.trace(covar) * np.eye(dataDim)
    return firstPart + secondPart
    

def tUEye(covar, dataDim, sampleSize):
    coeff = float(min(1, 4*(dataDim**2-1))/((sampleSize-dataDim-2)*dataDim**2))
    result = coeff * uEye(covar, dataDim)**(1.0/dataDim)
    return result

def uEye(covar, dataDim):
    result = dataDim * np.linalg.det(covar)**(1.0/dataDim)
    result = result / np.trace(covar)
    return result

def splitTrainDat(train, groupNum):
    group1 = [] 
    group2 = [] 
    group3 = [] 
    for i in map(int, train):
        if groupNum[i] == 1:
            group1.append(i)
        elif groupNum[i] == 2:
            group2.append(i)
        else: 
            group3.append(i)
    return (group1, group2, group3)
    

def classify(dist1, dist2, dist3):
    groupNum = 1
    if dist2 <= min(dist1, dist3):
        groupNum = 2
    if dist3 <= min(dist1, dist2):
        groupNum = 3
    return groupNum

def qdf(data, sampleCov, sampleMean):
    distance = math.log(np.linalg.det(np.matrix(sampleCov))) + \
        (data-sampleMean).T.dot(np.linalg.inv(np.matrix(sampleCov))).dot(data-sampleMean)
    return distance

rep = 500
dataDim = 10 
totalClass = 3
sampleSize = 7 
fold = 10
testMaxDim = 10
testSize = int(totalClass*sampleSize / fold)
trainSize = sampleSize - testSize
totalCer = np.zeros((rep, testMaxDim))

cov1 = np.eye(10)
cov2 = np.eye(10) + np.ones((10,10))
cov3 = np.eye(10) + np.ones((10, 10))
cov3[2, :] = np.zeros(10)
cov3[:, 2] = np.zeros(10)
cov3[2,2] = 10

mean1 = np.array([-1.43, -0.66, -0.94, 0.31, -0.19, 0.89, 0.25, -0.34, 1.25, -1.6])
mean2 = np.array([-0.43, 0.34, 0.06, 1.31, 0.81, 1.89, 1.25, 0.66, 2.25, -0.6])
mean3 = np.array([0.57, 1.34, 1.06, 2.31, 1.81, 2.89, 2.25, 1.66, 3.25, 0.4])

for dim in range(1, testMaxDim+1):
    cer = [] #initialize error rate to be empty list
