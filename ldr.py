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
    return np.asscalar(distance)

# Initialize experiment parameters
totalRep = 500
dataDim = 10 
totalClass = 3
trainSize = 7 
testMaxDim = 6 
testSize = 5000
totalCer = np.zeros((totalRep, testMaxDim))

# Generate training data
cov1 = np.eye(10)
cov2 = np.eye(10) + np.ones((10,10))
cov3 = np.eye(10) + np.ones((10, 10))
cov3[2, :] = np.zeros(10)
cov3[:, 2] = np.zeros(10)
cov3[2,2] = 10

mean1 = np.array([-1.43, -0.66, -0.94, 0.31, -0.19, 0.89, 0.25, -0.34, 1.25, -1.6])
mean2 = np.array([-0.43, 0.34, 0.06, 1.31, 0.81, 1.89, 1.25, 0.66, 2.25, -0.6])
mean3 = np.array([0.57, 1.34, 1.06, 2.31, 1.81, 2.89, 2.25, 1.66, 3.25, 0.4])

# Generate test data
aggregateErrorRate = np.zeros((totalRep, testMaxDim))
for rep in range(1, totalRep+1): 
    test1 = np.random.multivariate_normal(mean1, cov1, size = testSize)
    test2 = np.random.multivariate_normal(mean2, cov2, size = testSize)
    test3 = np.random.multivariate_normal(mean3, cov3, size = testSize)
    test = np.concatenate((test1, test2, test3), axis=0)
    groupNum = np.concatenate((np.ones(testSize), 2*np.ones(testSize), 3*np.ones(testSize)), axis =
            0)
    
    train1 = np.random.multivariate_normal(mean1, cov1, size=trainSize)
    train2 = np.random.multivariate_normal(mean2, cov2, size=trainSize)
    train3 = np.random.multivariate_normal(mean3, cov3, size=trainSize)
    
    sampleMean1 = np.mean(train1, axis=0).reshape(-1, 1)
    sampleMean2 = np.mean(train2, axis=0).reshape(-1, 1) 
    sampleMean3 = np.mean(train3, axis=0).reshape(-1, 1) 
    
    sampleCov1 = np.cov(train1.T)
    sampleCov2 = np.cov(train2.T)
    sampleCov3 = np.cov(train3.T)
    
    M1 = np.concatenate((sampleMean2-sampleMean1, sampleMean3-sampleMean2, sampleCov2-sampleCov1, sampleCov3-sampleCov1), axis=1)
    F, u, v = np.linalg.svd(M1, full_matrices=False)
    dataDim = 6#min(trainSize, math.floor(0.95*np.size(F, 1)))
    F = F[:, 0:dataDim] # Get the minimum of 95% of column and 7
    
    train1 = train1.dot(F)
    train2 = train2.dot(F)
    train3 = train3.dot(F)
    test = test.dot(F)
    
    sampleMean1 = np.mean(train1, axis=0).reshape(-1, 1)
    sampleMean2 = np.mean(train2, axis=0).reshape(-1, 1)
    sampleMean3 = np.mean(train3, axis=0).reshape(-1, 1)
    
    sampleCov1 = np.cov(train1.T)
    sampleCov2 = np.cov(train2.T)
    sampleCov3 = np.cov(train3.T)
    
    sTildeInv1 = sTildeInv(sampleCov1, dataDim, trainSize) 
    sTildeInv2 = sTildeInv(sampleCov2, dataDim, trainSize) 
    sTildeInv3 = sTildeInv(sampleCov3, dataDim, trainSize) 
    M2 = np.column_stack((sTildeInv2.dot(sampleMean2)-sTildeInv1.dot(sampleMean1),\
    sTildeInv3.dot(sampleMean3)-sTildeInv1.dot(sampleMean1), sampleCov2-sampleCov1,\
    sampleCov3-sampleCov1))
    Fhat,v,d = np.linalg.svd(M2)
    
    for dim in range(1, testMaxDim+1):
        F = Fhat[:, 0:dim] 
        reduceTrain1 = train1.dot(F)
        reduceTrain2 = train2.dot(F)
        reduceTrain3 = train3.dot(F)
        reduceTest = test.dot(F)
    
        sampleMean1 = np.mean(reduceTrain1, axis=0).reshape(-1, 1)
        sampleMean2 = np.mean(reduceTrain2, axis=0).reshape(-1, 1)
        sampleMean3 = np.mean(reduceTrain3, axis=0).reshape(-1, 1)
        
        sampleCov1 = np.cov(reduceTrain1.T)
        sampleCov2 = np.cov(reduceTrain2.T)
        sampleCov3 = np.cov(reduceTrain3.T)
    
        misclassify = 0
        for j in range(testSize):
            dist1 = qdf(reduceTest[j, :].reshape(-1,1), sampleCov1, sampleMean1)
            dist2 = qdf(reduceTest[j, :].reshape(-1,1), sampleCov2, sampleMean2)
            dist3 = qdf(reduceTest[j, :].reshape(-1,1), sampleCov3, sampleMean3)
            if classify(dist1, dist2, dist3) != groupNum[j]:
                misclassify += 1
        errorRate = float(misclassify) / testSize 
        aggregateErrorRate[rep-1, dim-1] = errorRate

np.save('dataFile', aggregateErrorRate)
