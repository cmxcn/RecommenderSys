# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:06:34 2015

@author: Mengxiang Chen
"""
import pickle
def trainDataPreprocessing(inputFile, outputFile, userIDFile, itemIDFile):
    #change userID and movieID into 0-n numbers

    items = dict()
    users = dict()
    userCount = 0
    itemCount = 0
    ratingCount = 0
    trainDataFile = open(outputFile, 'w')

    with open(inputFile) as inFile:
        for line in inFile:
            line = line.strip().split()
            u = int(line[0])
            i = int(line[1])
            r = float(line[2])
            if u in users:
                uid = users[u]
            else:
                uid = users[u] = userCount
                userCount+= 1
            if i in items:
                iid = items[i]
            else:
                iid = items[i] = itemCount
                itemCount += 1
            ratingCount += 1
            trainDataFile.write('{0} {1} {2}\n'.format(uid, iid, r))

    trainDataFile.close()
    
    userIDF = open(userIDFile, 'wb')
    pickle.dump(userCount, userIDF)
    pickle.dump(users, userIDF)
    userIDF.close()
    
    
    itemIDF = open(itemIDFile, 'wb')
    pickle.dump(itemCount, itemIDF)
    pickle.dump(items, itemIDF)
    itemIDF.close()
    
    return userCount, itemCount, ratingCount
    
    
if __name__=="__main__":
    inputFile = "ml-100k/u2.base"
    outputFile = "trainData"
    userIDFile = "userID"
    itemIDFile = "itemID"
    userCount, itemCount, ratingCount = trainDataPreprocessing(inputFile, outputFile, userIDFile, itemIDFile)
    print(userCount, itemCount, ratingCount)