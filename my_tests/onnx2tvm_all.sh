#!/bin/bash

ONNX_DIR=$1
DEST_DIR=$2

python3 onnx2tvm.py ${ONNX_DIR}/arcfaceresnet100-8.onnx ${DEST_DIR}/arcfaceresnet100-8_cuda_llis-pack.so cuda_llis 1 3 112 112
python3 onnx2tvm.py ${ONNX_DIR}/ultraface320.onnx ${DEST_DIR}/ultraface320_cuda_llis-pack.so cuda_llis 1 3 240 320
python3 onnx2tvm.py ${ONNX_DIR}/densenet121.onnx ${DEST_DIR}/densenet121_cuda_llis-pack.so cuda_llis 1 3 224 224
python3 onnx2tvm.py ${ONNX_DIR}/inception_v3.onnx ${DEST_DIR}/inception_v3_cuda_llis-pack.so cuda_llis 1 3 224 224
python3 onnx2tvm.py ${ONNX_DIR}/mnist-8.onnx ${DEST_DIR}/mnist-8_cuda_llis-pack.so cuda_llis 1 1 28 28
#python3 onnx2tvm.py ${ONNX_DIR}/gpt2.onnx ${DEST_DIR}/gpt2_cuda_llis-pack.so cuda_llis 1 3 224 224


