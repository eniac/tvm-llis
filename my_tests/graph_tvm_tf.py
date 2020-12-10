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

def main():
    module = tf_op.OpModule("mnist-8-cuda-pack.so")
    
    #input = tf.placeholder("float32", shape=[28 * 28])
    
    feed_dict = {
    }  
  
    run_ = module.func("run", output_shape=[0], output_dtype="float")
    set_input_ = module.func("set_input", output_shape=[0], output_dtype="float")
    get_output_ = module.func("get_output", output_shape=[0], output_dtype="float")
  
    @tf.function
    def run():
        run_(tf.constant([]))

    @tf.function
    def set_input(x):
        set_input_(x)

    @tf.function
    def get_output():
        return get_output_()
  
    #x = tf.constant([x for x in range(28*28)])
    #set_input(x)
    #run()
    with tf.device("/gpu:0"):
        print(get_output())


if __name__ == "__main__":
    main()

