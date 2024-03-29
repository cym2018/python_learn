# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Fashion-MNIST dataset."""

import gzip
import os

import numpy as np

from keras.src.utils.data_utils import get_file

# isort: off


def load_data():
    paths = [
        "/home/yuming/repo/fashion-mnist/data/fashion/train-labels-idx1-ubyte.gz",
        "/home/yuming/repo/fashion-mnist/data/fashion/train-images-idx3-ubyte.gz",
        "/home/yuming/repo/fashion-mnist/data/fashion/t10k-labels-idx1-ubyte.gz",
        "/home/yuming/repo/fashion-mnist/data/fashion/t10k-images-idx3-ubyte.gz",
    ]

    with gzip.open(paths[0], "rb") as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], "rb") as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(
            len(y_train), 28, 28
        )

    with gzip.open(paths[2], "rb") as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], "rb") as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(
            len(y_test), 28, 28
        )

    return (x_train, y_train), (x_test, y_test)

