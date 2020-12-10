import tvm
from tvm import relay

import numpy as np

from tvm.contrib.download import download_testdata

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.jit import ScriptModule

import sys

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        print(self.conv(x).shape)
        return self.conv(x)

model = ConvNet().to(device)
model = model.eval()

size = int(sys.argv[1])

input_shape = [1, 1, size, size]
input_data = torch.randn(input_shape, dtype=torch.float32).to(device)
scripted_model = torch.jit.trace(model, input_data).eval()

input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

target = "cuda_kelvin"
target_host = "c"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, target_host=target_host, params=params)
    lib.export_library("pytorch_conv_{}-pack.so".format(size))
