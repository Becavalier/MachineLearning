#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# Silence warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# Create a constant
matrix1 = tf.constant([3, 3], shape = [1, 2])
matrix2 = tf.constant([2, 2], shape = [2, 1])

# Multiplies matrix a by matrix b, producing a * b.
product = tf.matmul(matrix1, matrix2)

# There are three ops above

# Create a session
sess = tf.Session()

# Indicate that you want to take back the 'product' variable as a result
result = sess.run(product)

print('The result is: ', result)

# End up session
sess.close()



# Variables
x = tf.Variable([1.0, 2.0])
y = tf.constant([2.0, 1.0], shape = [2, 1])
z = tf.constant([3.0, 3.0], shape = [2, 1])

# Initialization of one variable
init_op = tf.global_variables_initializer()

sub = tf.subtract(x, y)
abs = tf.abs(tf.add(y, z))
square = tf.square(z)

with tf.Session() as sess:
    sess.run(init_op)
    abs, square = sess.run([abs, square])
    print(abs)
    print(square)