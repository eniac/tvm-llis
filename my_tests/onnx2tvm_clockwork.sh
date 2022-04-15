ONNX_DIR=$1
shift
PRECONVERT_DIR=$1
shift
POSTCONVERT_DIR=$1
shift
CONVERTER_PATH=$1
shift
name=$1
shift

python3 onnx2tvm_clockwork.py "${ONNX_DIR}/${name}.onnx" "${PRECONVERT_DIR}/${name}" cuda $@
mkdir -p "${POSTCONVERT_DIR}/${name}"
"${CONVERTER_PATH}" -o "${POSTCONVERT_DIR}/${name}" \
                    1 "${PRECONVERT_DIR}/${name}.1" \
                    2 "${PRECONVERT_DIR}/${name}.2" \
                    4 "${PRECONVERT_DIR}/${name}.4" \
                    8 "${PRECONVERT_DIR}/${name}.8" \
                    16 "${PRECONVERT_DIR}/${name}.16"
