#!/usr/bin/env python3
# -*- coding: utf-8 -*-

" KNN - Machine Learning "

__author__ = 'YHSPY'

from numpy import *
import operator
import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

# Cloumns setting
features = ('Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings')
classification_feature = 'Rings'
classification_number = 29


def features_normalization (features):
    row_count = features.shape[0]
    min = features.min(axis = 0)
    max = features.max(axis = 0)

    step = max - min

    normalization_matrix = zeros(features.shape)
    normalization_matrix = features - tile(min, [row_count, 1])
    normalization_matrix = normalization_matrix / tile(step, [row_count, 1])
    return normalization_matrix, step, min


def process_input_samples (path):
    samples_file_handler = open(path, mode='r')
    formatted_samples_class = []
    sample_index = 0

    samples_all_lines = samples_file_handler.readlines()
    samples_all_lines_len = len(samples_all_lines)
    samples_features_len = len(features)

    # Generate a matrix with zero
    samples_all_matrix = zeros([samples_all_lines_len, samples_features_len - 1])
    for line in samples_all_lines:
        line = line.strip()
        sample_features_list = line.split(',')
        samples_all_matrix[sample_index, :] = sample_features_list[:samples_features_len - 1]
        formatted_samples_class.append(int(sample_features_list[-1]))
        sample_index += 1
    return samples_all_matrix, formatted_samples_class



def handle_samples (path):
    samples_all_matrix, formatted_samples_class = process_input_samples(path)
    normalization_matrix, step, min = features_normalization(samples_all_matrix)
    return normalization_matrix, step, min, formatted_samples_class


def handle_testcase_samples(path, step, min):
    samples_all_matrix, classification_initial_list  = process_input_samples(path)
    # Normalization according to previous training set
    row_count = samples_all_matrix.shape[0]
    normalization_matrix = zeros(samples_all_matrix.shape)
    normalization_matrix = samples_all_matrix - tile(min, [row_count, 1])
    normalization_matrix = normalization_matrix / tile(step, [row_count, 1])

    return normalization_matrix


parser = argparse.ArgumentParser (description='KNN - YHSPY')
parser.add_argument('--samples', help = 'Input the path of sample file for KNN algorithm')
parser.add_argument('--test', help = 'Input the path of predict samples file.')
parser.add_argument('--tf', help = 'Use tensorflow as an analysis tool', action = 'store_true')

# Extract input parameters
samples_path = parser.parse_args().samples
testcase_samples_path = parser.parse_args().test
use_tensorflow = parser.parse_args().tf

if os.path.exists(samples_path):
    if use_tensorflow:
        samples_all_matrix, step, min, formatted_samples_class = handle_samples(samples_path)
        samples_all_testcase_matrix = handle_testcase_samples(testcase_samples_path, step, min)

        features_count = samples_all_matrix.shape[1]
        xtr = tf.placeholder("float", [None, features_count])
        xte = tf.placeholder("float", [features_count])

        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.add(xtr, tf.negative(xte))), axis = 1))

        pred = tf.argmin(distance, 0)

        # Initial variables
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)

            for i in range(len(samples_all_testcase_matrix)):
                predict_index = sess.run(pred, feed_dict = {xtr: samples_all_matrix, xte: samples_all_testcase_matrix[i, :]})
                print("nn_indext:", predict_index)
                print("Ytr[nn_index]:", formatted_samples_class[predict_index])


    else:
        samples_all_matrix, step, min, formatted_samples_class = handle_samples(samples_path)
        samples_all_testcase_matrix = handle_testcase_samples(testcase_samples_path, step, min)
        row_count = samples_all_matrix.shape[0]

        for testcasae in samples_all_testcase_matrix:
            diffMat = tile(testcasae, [row_count, 1]) - samples_all_matrix
            # argsort return the index of elements after sorted
            distance = ((diffMat ** 2).sum(axis = 1)) ** 0.5
            distanceSorted = distance.argsort()
            voteCount = {}

            # General k
            for i in range(int(len(samples_all_matrix) ** 0.5)):
                voteLable = formatted_samples_class[distanceSorted[i]]
                voteCount[voteLable] = voteCount.get(voteLable, 0) + (1 / distance[distanceSorted[i]])
            # itermitems return an iterator used for dict
            sortedVotes = sorted(voteCount.items(), key = operator.itemgetter(1), reverse = True)
            print (sortedVotes[0][0])
else:
    raise Exception('[Exception] Invalid path of input samples.')
