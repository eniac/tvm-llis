import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
import sys

def change_input_dim(model, dim):
    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim 
    inputs = model.graph.input
    for input in inputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_param = str(dim)

model_path = sys.argv[1] # path to the onnx model
output_path = sys.argv[2] # path to output the tvm-compiled model
target = sys.argv[3] # cuda or cuda_llis
input_dims = [int(x) for x in sys.argv[4:]] # e.g., `1 3 224 224` for mobilenet

target_host = "llvm"

onnx_model = onnx.load(model_path)

for batch_size in [1, 2, 4, 8, 16]:
    change_input_dim(onnx_model, batch_size)
    input_dims[0] = batch_size

    input_names_all = [node.name for node in onnx_model.graph.input]
    input_initializer =  [node.name for node in onnx_model.graph.initializer]
    input_names = list(set(input_names_all)  - set(input_initializer))

    input_name = input_names[0]

    shape_dict = {input_name: tuple(input_dims)}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    opt_level = 3
    with relay.build_config(opt_level=opt_level):
        graph, lib, params = relay.build(mod, target, target_host, params=params)

    lib.export_library(output_path + '.' + str(batch_size) + '.so')

    with open(output_path + '.' + str(batch_size) + '.json', 'w') as f:
        f.write(graph)

    with open(output_path + '.' + str(batch_size) + '.params', 'wb') as f:
        f.write(relay.save_param_dict(params))

