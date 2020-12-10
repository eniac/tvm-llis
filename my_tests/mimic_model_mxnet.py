from mxnet.gluon import HybridBlock, nn, rnn
import mxnet as mx

import tvm
from tvm import te
import tvm.relay as relay

model = nn.HybridSequential()
with model.name_scope():
    model.add(rnn.LSTM(180, 2, input_size=18))
    model.add(nn.Dense(2))

model.initialize(ctx=mx.gpu(0))
model.hybridize()
#init_states = (F.zeros((2, 1, 180), ctx=mx.gpu(0)), F.zeros((2, 1, 180), ctx=mx.gpu(0)))
print(model(mx.nd.zeros((1, 1, 18), ctx=mx.gpu(0))))

opt_level = 3
target = "cuda_llis"
target_host = "c"
shape_dict = {"data": (1, 1, 18)}
mod, params = relay.frontend.from_mxnet(model, shape_dict)
with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, target_host, params=params)

lib.export_library("mimic-{}-pack.so".format(target))

