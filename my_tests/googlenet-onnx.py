import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
import time

model_url = 'https://media.githubusercontent.com/media/onnx/models/master/vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.onnx'

model_path = download_testdata(model_url, "googlenet-9.onnx", module="onnx")

onnx_model = onnx.load(model_path)

input_name = "data_0"
shape_dict = {input_name: (1, 3, 224, 224)}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

opt_level = 3
target_host = "c"
#target = "cuda_llis"
target = "cuda"
with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, target_host, params=params)
lib.export_library("googlenet-9-{}-pack.so".format(target))
