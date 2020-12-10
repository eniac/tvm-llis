import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata

model_url = 'https://media.githubusercontent.com/media/onnx/models/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx'

model_path = download_testdata(model_url, "mobilenetv2-7.onnx", module="onnx")

onnx_model = onnx.load(model_path)

#target_host = "llvm"
target_host = "c"

input_name = "data"
shape_dict = {input_name: (1, 3, 224, 224)}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

opt_level = 3
#target = tvm.target.cuda()
target = "cuda"
#target = "cuda_llis"
#target = tvm.target.cuda_llis()
with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, target_host, params=params)

#from tvm.contrib import cc

#module = lib.module

#module._collect_dso_modules()[0].save("mnist-8.o")
#module._collect_dso_modules()[0].save("mnist-8.c")
#cc.create_shared("mnist-8.so", ["mnist-8.o"])

lib.export_library("mobilenetv2-7-{}-pack.so".format(target))
