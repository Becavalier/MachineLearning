#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
import csv

HORATIO = 0.1

def printMatplot(dataMat, dataLabels):
    plt.figure(figsize=(9, 6))
    plt.scatter(dataMat[:,0], dataMat[:,1], s=30, c='red', marker='x', alpha=1)
    plt.show()

def createDataSet():
    group = array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1]
    ])
    labels = ['A', 'A', 'B', 'B']
    printMatplot(group, labels)
    return group, labels

def createDataSetFromFile(path):
    group, labels = [], []
    with open(path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            group.append(list(map(lambda x:float(x), row[:-1])))
            labels.append(row[-1])
    return array(group), labels

# 高斯函数
def gaussian(dist, a=1, b=0, c=0.3):
    return a * math.e ** (-(dist - b) ** 2 / (2 * c ** 2))

# 归一化；
# *有时，归一化会消除数据重要的特征差异，从而导致精度下降。
# newValue = (oldValue - mins) / (max - min)
def autoNorm(dataSet):
    # 每列的最小值；
    minVals = dataSet.min(0)
    # 每列的最大值；
    maxVals = dataSet.max(0)
    # 差值；
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 每行减去最小值；
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 每行除以差值；
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def classify (intX, dataSet, labels, k):
    # dataSet, ranges, minVals = autoNorm(dataSet)
    # 行数；
    dataSetSize = dataSet.shape[0]
    # 复制多行，求差；
    diffMat = tile(intX, (dataSetSize, 1)) - dataSet
    # 平方；
    sqDiffMat = diffMat ** 2
    # 求和；
    sqDistances = sqDiffMat.sum(axis=1)
    # 开根号（欧氏距离）；
    distances = sqDistances ** 0.5
    # 正序排序，返回索引；
    sortedDistIndicies = distances.argsort()

    classCount = {}
    # 选择最近的 k 个数据，统计各标签的出现次数；
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1


    # 降序排序，选择最近的一个标签作为结果；
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


if __name__ == "__main__":
    group, labels = createDataSetFromFile("./data.csv")
    # 行数；
    dataSetSize = group.shape[0]
    # 测试集数量；
    numTestCount = int(dataSetSize * HORATIO)
    errTestCount = 0
    for i in range(numTestCount):
        # http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
        result = classify(group[i], group[numTestCount:], labels[numTestCount:], 3)
        if result != labels[i]:
            errTestCount += 1
    print("The total error rate is: %f" % (errTestCount / numTestCount))
