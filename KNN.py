#!/usr/bin/env python3
# -*- coding: utf-8 -*-

" KNN - Machine Learning "

__author__ = 'YHSPY'

from numpy import *
import operator
import argparse
import os

def process_input_samples (path):
    samples_file_handler = open(path, mode='r')

    

def handle_samples (path):



def handle_samples_with_tensorflow (path):


parser = argparse.ArgumentParser (description='KNN - YHSPY')
parser.add_argument('--samples', help = 'Input the path of sample file for KNN algorithm')
parser.add_argument('--ts', help = 'Use tensorflow as an analysis tool', action = 'store_true')

" Extract input parameters "
samplesPath = parser.parse_args().samples
ifUseTensorflow = parser.parse_args().ts

if os.path.exists(samplesPath):
    if ifUseTensorflow:
        handle_samples_with_tensorflow()
    else:
        handle_samples(samplesPath)
else:
    raise Exception('[Exception] Invalid path of input samples.')



