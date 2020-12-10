#!/usr/bin/env python

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tensorflow as tf
import sys

from tvm.contrib import tf_op
# import tf_op

import numpy as np

class Model(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[10, 20], dtype=tf.float32)])
    def __call__(self, x):
        return self.run(x, self.y, self.z)
    
    def __init__(self, run, y, z):
        super(Model, self).__init__()

        self.run = run
        self.y = tf.Variable(y, name='w1')
        self.z = tf.Variable(z, name='w2')

def main():
    module = tf_op.OpModule("fully_connected-cuda-pack.so")
    
    run = module.func("run", output_shape=[10, 25], output_dtype="float")
  
    x = np.random.rand(10, 20).astype(np.float32)
    y = np.random.rand(20, 15).astype(np.float32)
    z = np.random.rand(15, 25).astype(np.float32)

    model = Model(run, y, z)

    with tf.device("/gpu:0"):
        print(model(x))

    print(np.matmul(np.matmul(x, y), z))

    tf.saved_model.save(model, 'fully_connected_tvm_tf_model')

if __name__ == "__main__":
  main()

