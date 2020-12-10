from matmul import matmul
from matmul import matmul2
import tvm
from tvm import te

target = "cuda"

n = 10

X = te.placeholder((10, 20), name='X')
Y = te.placeholder((20, 15), name='Y')
Z = te.placeholder((15, 25), name='Z')

D = matmul2(X, Y)
C = matmul2(D, Z)

s = te.create_schedule(C.op)
bx, tx = s[C].split(C.op.axis[0], factor=64)
s[C].bind(bx, te.thread_axis("blockIdx.x"))
s[C].bind(tx, te.thread_axis("threadIdx.x"))
bx, tx = s[D].split(D.op.axis[0], factor=64)
s[D].bind(bx, te.thread_axis("blockIdx.x"))
s[D].bind(tx, te.thread_axis("threadIdx.x"))
#print(tvm.lower(s, [A, B], simple_mode=True))
mod = tvm.build(s, [X, Y, Z, C], target, name="run")
mod.export_library("fully_connected-{}-pack.so".format(target));

