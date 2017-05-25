#!/usr/bin/env python3
# -*- coding: utf-8 -*-

" KNN - Machine Learning "

__author__ = 'YHSPY'

from numpy import *
import operator
import argparse
import os

" Cloumns setting"
features = ('Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings')
classification_feature = 'Rings'
classification_number = 29
classification_k = 100

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

    " Generate a matrix with zero"
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
    samples_all_matrix, step, min = features_normalization(samples_all_matrix)


def handle_samples_with_tensorflow (path):
    process_input_samples(path)

parser = argparse.ArgumentParser (description='KNN - YHSPY')
parser.add_argument('--samples', help = 'Input the path of sample file for KNN algorithm')
parser.add_argument('--input', help = 'Input the path of predict samples file.')
parser.add_argument('--ts', help = 'Use tensorflow as an analysis tool', action = 'store_true')

" Extract input parameters "
samples_path = parser.parse_args().samples
input_samples_path = parser.parse_args().input
if_use_tensorflow = parser.parse_args().ts

if os.path.exists(samples_path):
    if if_use_tensorflow:
        handle_samples_with_tensorflow(samples_path)
    else:
        handle_samples(samples_path)
else:
    raise Exception('[Exception] Invalid path of input samples.')



