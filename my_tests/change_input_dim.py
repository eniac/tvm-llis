# Source: https://github.com/onnx/onnx/issues/2182#issuecomment-513888258

import onnx
import sys

def change_input_dim(model, dim):
    # Use some symbolic name not used for any other dimension
    #sym_batch_dim = "N"
    # or an actal value
    #actual_batch_dim = 4 

    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim 
    inputs = model.graph.input
    for input in inputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]
        # update dim to be a symbolic value
        # dim1.dim_param = dim
        # or update it to be an actual value:
        dim1.dim_value = str(dim)

infile = sys.argv[1]
outfile = sys.argv[2]
dim = int(sys.argv[3])

model = onnx.load(infile)
change_input_dim(model, dim)
onnx.save(model, outfile)

