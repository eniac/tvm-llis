import sys
import time

import numpy as np

import onnx
from onnx_tf.backend import prepare

path = sys.argv[1]
num_iters = int(sys.argv[2])
input_name = sys.argv[3]
input_shape = [int(x) for x in sys.argv[4:]]

input_data = tvm.nd.array(np.random.uniform(size=input_shape).astype("float32"), ctx=ctx)

onnx_model = onnx.load(path)
tf_model = prepare(onnx_model)

start_time = time.time()

for i in range(num_iters):
    output = tf_model.run(input)
