#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# 最优化算法；

# Sigmoid 函数，fx = 1/(1 + e ** -x)；
# Sigmoid 函数输入：x = w0x0 + w1x1 + w2x2 + ... + wnxn

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('./data.csv')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1 + np.exp(-inX))

def gradAscent(dataMathIn, classLabels):
    # Trans to matrix
    dataMatrix = np.mat(dataMathIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix) # (100, 3)
    # Foot step
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))

    for k in range(maxCycles):
        # (100, 1)
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        # 极大似然估计；
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    print(gradAscent(dataMat, labelMat))
    # print(dataSet)