#!/bin/bash

ONNX_DIR=$1
PRECONVERT_DIR=$2
POSTCONVERT_DIR=$3
CONVERTER_PATH=$4

bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} mnist-8 1 1 28 28
bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} arcfaceresnet100-8 1 3 112 112
bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} ultraface320 1 3 240 320
bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} densenet121 1 3 224 224
bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} inception_v3 1 3 224 224
#bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} gpt2 1 3 224 224

