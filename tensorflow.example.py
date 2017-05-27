#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

# Load MNIST
from tensorflow.examples.tutorials.mnist import input_data

# Read dataset
'''
Datasets(train=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x114025fd0>, validation=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object
at 0x10b667c50>, test=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x10b664780>)
'''
mnist = input_data.read_data_sets("MNIST_Data/data/", one_hot=True)

Xtr, Ytr = mnist.train.next_batch(5000)
Xte, Yte = mnist.test.next_batch(200)

xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

distance = tf.sqrt(tf.reduce_sum(tf.square(tf.add(xtr, tf.negative(xte))), axis=1))

pred = tf.argmin(distance, 0)

accuracy = 0.

init = tf.initialize_all_variables()

# Launch  graph
with tf.Session() as sess:
    sess.run(init)

    for i in range(len(Xte)):
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        print("nn_indext:", nn_index)
        print("Ytr[nn_index]:", Ytr[nn_index])
        print("np.argmax(Ytr[nn_index]):", np.argmax(Ytr[nn_index]))
        print ("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), "True Class:", np.argmax(Yte[i]))

        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1. / len(Xte)
    print ("Done!")
    print ("Accuracy:", accuracy)