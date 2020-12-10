import sys
import time

import numpy as np

from tvm import relay
import tvm
from tvm import te
from tvm.contrib import graph_runtime

path = sys.argv[1]
num_iters = int(sys.argv[2])
input_name = sys.argv[3]
input_shape = [int(x) for x in sys.argv[4:]]

ctx = tvm.gpu()

loaded_lib = tvm.runtime.load_module(path)
input_data = tvm.nd.array(np.random.uniform(size=input_shape).astype("float32"), ctx=ctx)

module = graph_runtime.GraphModule(loaded_lib["default"](ctx))

input_dict = {
    input_name: input_data
}

start_time = time.time()
for i in range(num_iters):
    module.run(**input_dict)
    out_deploy = module.get_output(0).asnumpy()
end_time = time.time()

print((end_time - start_time) * 1000000 / num_iters)
