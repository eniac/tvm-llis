#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

int main(int argc, char** argv) {
    DLContext ctx_gpu{kDLGPU, 0};
    DLContext ctx_cpu{kDLCPU, 0};

    //tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile("myadd.so");
    //tvm::runtime::Module mod_dev = tvm::runtime::Module::LoadFromFile("myadd.ptx");
    //mod.Import(mod_dev);
    tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile("myadd_pack.so");

    tvm::runtime::PackedFunc myadd = mod.GetFunction("myadd");

    tvm::runtime::NDArray A = tvm::runtime::NDArray::Empty({10}, DLDataType{kDLFloat, 32, 1}, ctx_cpu);
    tvm::runtime::NDArray B = tvm::runtime::NDArray::Empty({10}, DLDataType{kDLFloat, 32, 1}, ctx_cpu);
    tvm::runtime::NDArray C = tvm::runtime::NDArray::Empty({10}, DLDataType{kDLFloat, 32, 1}, ctx_cpu);

    tvm::runtime::NDArray A_dev = tvm::runtime::NDArray::Empty({10}, DLDataType{kDLFloat, 32, 1}, ctx_gpu);
    tvm::runtime::NDArray B_dev = tvm::runtime::NDArray::Empty({10}, DLDataType{kDLFloat, 32, 1}, ctx_gpu);
    tvm::runtime::NDArray C_dev = tvm::runtime::NDArray::Empty({10}, DLDataType{kDLFloat, 32, 1}, ctx_gpu);

    for (int i = 0; i < 10; ++i) {
        reinterpret_cast<float*>(A->data)[i] = i + 1;
        reinterpret_cast<float*>(B->data)[i] = i * 2 + 2;
    }

    A_dev.CopyFrom(A);
    B_dev.CopyFrom(B);

    //set_input("A", A);
    //set_input("B", B);
    //run();
    //get_output(0, C);

    myadd(A_dev, B_dev, C_dev);

    C.CopyFrom(C_dev);

    for (int i = 0; i < 10; ++i) {
        printf("%f\n", reinterpret_cast<float*>(C->data)[i]);
    }
}

