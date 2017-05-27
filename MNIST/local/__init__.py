# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

import struct
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def save_local_images(filename, count):
    binfile = open(filename, 'rb')
    buf = binfile.read()

    index = 0
    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    for image in range(0, count):
        im = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')

        im = np.array(im, dtype='uint8')
        im = im.reshape(28, 28)
        im = Image.fromarray(im)
        im.save('part_of_sample_images/train_%s.bmp' % image, 'bmp')


