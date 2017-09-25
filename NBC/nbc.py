#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import re

# 基于概率论的贝叶斯定理，适用于标称型（离散）数据；

# 基于贝努利模型；

def splitWords(text):
    regEx = re.compile('\\W*')
    listOfTokens = regEx.split(text)
    return [tok.lower() for tok in listOfTokens if len(tok) > 0]

def loadDataSet():
    postingList= [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmatian', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    # 类别向量, 1-bad, 0-normal；
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        # 并集；
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    # 与词汇表等长；
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('Missing word: %s' % word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix) # 6；
    numWords = len(trainMatrix[0]) # 32；
    pAbusive = sum(trainCategory) / float(numTrainDocs) # 3/6 = 0.5
    # 防止乘积均为0的情况；
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        # bad；
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 每一个单词在所有出现单词中的概率；
    # 取对数避免下溢出；
    # 两个概率向量；
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # pClass1 = pc1
    # pClass0 = 1 - pc1
    # ?
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    listOPosts, listClasses = loadDataSet()
    # 所有词的集合；
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0(trainMat, listClasses)

    testEntry = ['love', 'my', 'dalmatian']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(thisDoc)
    print(p0V)
    print(classifyNB(thisDoc, p0V, p1V, pAb))
