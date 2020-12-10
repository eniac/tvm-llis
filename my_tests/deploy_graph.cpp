#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

int main(int argc, char** argv) {
    DLContext ctx_gpu{kDLGPU, 0};
    DLContext ctx_cpu{kDLCPU, 0};

    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("mnist-8-pack.so");
    tvm::runtime::Module gmod = mod_factory.GetFunction("default")(ctx_gpu);
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");

    tvm::runtime::NDArray input = tvm::runtime::NDArray::Empty({1, 1, 28, 28}, DLDataType{kDLFloat, 32, 1}, ctx_cpu);
    tvm::runtime::NDArray output = tvm::runtime::NDArray::Empty({1, 10}, DLDataType{kDLFloat, 32, 1}, ctx_cpu);

    tvm::runtime::NDArray input_dev = tvm::runtime::NDArray::Empty({1, 1, 28, 28}, DLDataType{kDLFloat, 32, 1}, ctx_gpu);
    tvm::runtime::NDArray output_dev = tvm::runtime::NDArray::Empty({1, 10}, DLDataType{kDLFloat, 32, 1}, ctx_gpu);

    input_dev.CopyFrom(input);

    set_input("Input3", input_dev);
    run();
    get_output(0, output_dev);

    output.CopyFrom(output_dev);

    for (int i = 0; i < 10; ++i) {
        printf("%f\n", reinterpret_cast<float*>(output->data)[i]);
    }
}

