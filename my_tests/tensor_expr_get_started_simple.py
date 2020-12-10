from __future__ import absolute_import, print_function

import tvm
import tvm.testing
from tvm import te
import numpy as np

# Global declarations of environment.

tgt_host = "llvm"
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl, rocm
#tgt = "cuda"
tgt = "cuda_kelvin"

n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
print(type(C))

s = te.create_schedule(C.op)

bx, tx = s[C].split(C.op.axis[0], factor=64)

s[C].bind(bx, te.thread_axis("blockIdx.x"))
s[C].bind(tx, te.thread_axis("threadIdx.x"))

fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")

from tvm.contrib import cc

fadd.save("myadd.o")
#fadd.imported_modules[0].save("myadd.ptx")
fadd.imported_modules[0].save("myadd.ptx_kelvin")
#fadd.imported_modules[0].save("myadd.cu")
cc.create_shared("myadd.so", ["myadd.o"])

fadd.export_library("myadd_pack.so")
