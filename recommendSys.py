# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:12:11 2015

@author: Mengxiang Chen
"""
import pickle
import numpy as np

# encoding=utf-8

def biasedSVDPredict(miu, bu, bi, pu, qi):
    pu = np.array(pu)
    qi = np.array(qi)

    r = miu+bu+bi+sum(pu*qi)
    if r > 5:
        r = 5
    elif r < 1:
        r = 1
    return r

def validateModel(testFileName, userIDFileName, itemIDFileName, modelFileName):
    #calculate the RMSE

    userIDFile = open(userIDFileName, 'rb')
    userNum = pickle.load(userIDFile)
    users = pickle.load(userIDFile)
    userIDFile.close()
    itemIDFile = open(itemIDFileName, 'rb')
    itemNum = pickle.load(itemIDFile)
    items = pickle.load(itemIDFile)
    itemIDFile.close()
    modelFile = open(modelFileName, 'rb')
    miu = np.load(modelFile)
    bu = np.load(modelFile)
    bi = np.load(modelFile)
    pu = np.load(modelFile)
    qi = np.load(modelFile)

    testFile = open(testFileName)
    N = 0
    errSum = 0.0
    for line in testFile:
        line = line.strip().split()
        user = int(line[0])
        item = int(line[1])
        rui = float(line[2])
        if not user in users or not item in items:
            continue
        else:
            uid = users[user]
            iid = items[item] 
            y = biasedSVDPredict(miu, bu[uid], bi[iid], pu[uid], qi[iid])
        err = rui-y
        errSum += err**2
        N += 1
    rmse = (errSum/N)**0.5
    print("the RMSE is {0}\n".format(rmse))
    return rmse
    
def biasedSVDTrain(trainDataFileName, userIDFileName, itemIDFileName, configureFileName, modelFileName):
    #calculate bu, bi, pu, qi such that
    #rui = miu+bu+bi+pu*qi
    configureFile = open(configureFileName)
    #args in configure file: 
    #dimNum, learningRate, regulationCoef, iteration rounds
    line = configureFile.readline().split()
    dimNum = int(line[0])
    learningRate = float(line[1])
    regulationCoef = float(line[2])
    iterRounds = int(line[3])
    configureFile.close()
    print(dimNum, learningRate, regulationCoef, iterRounds, '\n')
    userIDFile = open(userIDFileName, 'rb')
    userNum = pickle.load(userIDFile)
    userIDFile.close()
    
    itemIDFile = open(itemIDFileName, 'rb')
    itemNum = pickle.load(itemIDFile)
    itemIDFile.close()
    
    print(userNum, itemNum, '\n')
    bu = np.zeros(userNum)
    bi = np.zeros(itemNum)
    pu = 0.1*np.random.random([userNum, dimNum])/(dimNum**0.5)
    qi = 0.1*np.random.random([itemNum, dimNum])/(dimNum**0.5)

    #first round calculate miu
    trainData = open(trainDataFileName)
    N = 0
    sumR = 0.0
    for line in trainData:
        line = line.strip().split()
        sumR += float(line[2])
        N += 1
    miu = sumR/N
    trainData.close()
    print("miu:", miu, '\n')
    
    print(miu, bu, bi, pu, qi)
    for i in range(iterRounds):
        print("{0} round\n".format(i))
        trainData = open(trainDataFileName)
        for line in trainData:
            line = line.strip().split()
            uid = int(line[0])
            iid = int(line[1])
            rui = float(line[2])
            y = miu + bu[uid] + bi[iid] + sum(pu[uid]*qi[iid])
            e = rui-y
            bu[uid] += learningRate*(e - regulationCoef*bu[uid])
            bi[iid] += learningRate*(e - regulationCoef*bi[iid])
            #note here we need to back up old pu for qi
            oldPu = pu[uid][:]
            pu[uid] += learningRate*(e*qi[iid] - regulationCoef*pu[uid])
            qi[iid] += learningRate*(e*oldPu - regulationCoef*qi[iid])
        trainData.close()
        
        SVDModelFile = open(modelFileName, 'wb')
        pickle.dump(miu, SVDModelFile)
        bu.dump(SVDModelFile)
        bi.dump(SVDModelFile)
        pu.dump(SVDModelFile)
        qi.dump(SVDModelFile)
        SVDModelFile.close()
        validateModel("ml-100k/u2.test", userIDFileName, itemIDFileName, modelFileName)
        
if __name__ == "__main__":
    trainData = "trainData"
    userIDs = "userID"
    itemIDs = "itemID"
    confs = "SVD.conf"
    modelF = "SVDModel"
    biasedSVDTrain(trainData, userIDs, itemIDs, confs, modelF)
    testFile = "ml-100k/u2.test"
    validateModel(testFile, userIDs, itemIDs, modelF)
        