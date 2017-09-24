#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import log
import operator
import pickle


# Based on ID3 algorithm（适用于标称型数据，即离散值）
'''
ID3 以信息熵的下降速度为选取测试属性的标准，即在每个节点选取还尚未被用来划分的具有最高信息增益的属性作为划分标准，然后继续这个过程，直到生成的决策树能完美分类训练样例。
'''

# 基于信息增益 - 划分数据集前后信息发生的变化
def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 计算信息熵；
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# 基于给定的特征键值来选择数据集；
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    # 基准信息熵；
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    # 遍历每一个特征；
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        # 每个特征对应的所有取值类型；
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 遍历每一个特征值；
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            # 熵加权；
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 新熵越小越好；
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def createTree(dataSet, labels):
    tLabels = labels[:]
    classList = [example[-1] for example in dataSet]
    # 所有类标签完全相同，分类结束；
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 标签用完，开始表决；
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择熵改变最大的特征进行分割，返回标签索引；
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = tLabels[bestFeat]
    myTree = {
        bestFeatLabel: {}
    }
    del(tLabels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 复制；
        subLabels = tLabels[:]
        # 递归；
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    # [0, 1]
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    # print(splitDataSet(dataSet, 0, 1))
    # print(chooseBestFeatureToSplit(dataSet))
    # shannonEnt = calcShannonEnt(dataSet)
    # print(shannonEnt)
    myTree = createTree(dataSet, labels)
    print(classify(myTree, labels, [1, 1]))
