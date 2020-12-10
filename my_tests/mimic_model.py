import tvm
from tvm import relay

import numpy as np

from tvm.contrib.download import download_testdata

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.jit import ScriptModule

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class NetworkApproxLSTM(nn.Module):
    def __init__(self, input_size=18, num_layers=2, window_size=10):
        super(NetworkApproxLSTM, self).__init__()

        self.input_size = input_size
        # remember window_size packets
        self.hidden_size = input_size * window_size
        self.num_layers = num_layers
        self.window_size = window_size
        self.output_size = 1

        # batch_first – If True, then the input and output tensors are provided
        # as (batch, seq, feature) or (batch, num_layers, hidden_size).
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                            batch_first = True) 

        self.linear = nn.Linear(self.hidden_size, 2)

    def init_hidden(self, batch_size = 1):
        h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                          dtype=torch.double).to(device)
        c_t = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                          dtype=torch.double).to(device)
        return (h_t, c_t)
    
    def forward(self, data, hidden_state):
        X = data.view(-1, self.window_size, self.input_size)
        batch_size = X.shape[0]
        lstm_out, hidden_state = self.lstm(X, hidden_state)

        output = self.linear(lstm_out[:,-1,:].view(batch_size, -1))

        return output, hidden_state

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x)

model = NetworkApproxLSTM(18, 2, 1).to(device)
model.init_hidden()
#model = TestNet().to(device)
model.double()
model = model.eval()

input_shape = [1, 1, 18]
#input_shape = [1, 1, 10, 10]
input_data = torch.randn(input_shape, dtype=torch.double).to(device)
scripted_model = torch.jit.trace(model.forward, (input_data, model.init_hidden())).eval()

input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
#mod, params = relay.frontend.from_pytorch(model, shape_list)

target = "cuda_kelvin"
target_host = "c"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, target_host=target_host, params=params)
    lib.export_library("mimic-pack.so")

