#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/cuda_kelvin_bench.h>

#include "tvmbench.h"

#include <cuda_runtime.h>

#include <iostream>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>
#include <thread>
#include <algorithm>

std::vector<double> latencies;

void func(int num_streams) {
    DLContext ctx_gpu{kDLGPU, 0};

    std::vector<tvm::runtime::NDArray> input_devs;
    std::vector<tvm::runtime::NDArray> output_devs;
    std::vector<tvm::runtime::Module> mod_factories;
    std::vector<tvm::runtime::Module> gmods;
    std::vector<tvm::runtime::PackedFunc> run_funcs(num_streams);

    CUDAKelvinBench* cuda_kelvin_bench = CUDAKelvinBench::get(num_streams);
    volatile unsigned* flags = cuda_kelvin_bench->get_flags();

    input_devs.resize(num_streams);
    output_devs.resize(num_streams);
    mod_factories.resize(num_streams);
    gmods.resize(num_streams);
    run_funcs.resize(num_streams);

    for (int i = 0; i < num_streams; ++i) {
        input_devs[i] = tvm::runtime::NDArray::Empty({1, 1, 28, 28}, DLDataType{kDLFloat, 32, 1}, ctx_gpu);
        output_devs[i] = tvm::runtime::NDArray::Empty({1, 10}, DLDataType{kDLFloat, 32, 1}, ctx_gpu);
        std::string path = "mnist-8-pack-";
        path += std::to_string(i);
        path += ".so";
        mod_factories[i] = tvm::runtime::Module::LoadFromFile(path);
        gmods[i] = mod_factories[i].GetFunction("default")(ctx_gpu);
        run_funcs[i] = gmods[i].GetFunction("run");
    }

    std::vector<std::chrono::time_point<std::chrono::steady_clock>> start_times;
    start_times.reserve(num_streams);

    auto very_start_time = std::chrono::steady_clock::now();

    for (int sid = 0; sid < num_streams; ++sid) {
        auto start_time = std::chrono::steady_clock::now();

        volatile unsigned* flag = &(flags[sid]);
        *flag = 0;
        run_funcs[sid]();

        start_times.push_back(start_time);
    }

    while (true) {
    //for (int i = 0; i < 9; ++i) {
        int sid = -1;
        do {
            for (int i = 0; i < num_streams; ++i) {
                //if (flags[i] == 49) {
                //if (flags[i] == 28) {
                if (flags[i] == 2) {
                    sid = i;
                    break;
                }
            }
        } while (sid == -1);

        //printf("sid: %d|\n", sid);

        volatile unsigned* flag = &(flags[sid]);

        auto cur_time = std::chrono::steady_clock::now();

        auto latency = std::chrono::duration<double, std::micro>(cur_time - start_times[sid]).count();
        auto time_elasped = std::chrono::duration<double, std::micro>(cur_time - very_start_time).count();

        if (time_elasped > 10000000) { // 10s
            latencies.push_back(latency);
        }

        if (time_elasped > 30000000) { // 30s
            return;
        }

        start_times[sid] = cur_time;

        *flag = 0;
        run_funcs[sid]();
    }
}

int main(int argc, char** argv) {
    int num_streams = atoi(argv[1]);
    const char* output_path = argv[2];

    func(num_streams);

    double throughput = latencies.size() / 20.0;

    std::sort(latencies.begin(), latencies.end());

    double mean = 0;

    for (double latency : latencies) {
        mean += (latency / (double)latencies.size());
    }
    
    double p50 = latencies[latencies.size() / 2];
    double p90 = latencies[latencies.size() * 0.90];
    double p95 = latencies[latencies.size() * 0.95];
    double p99 = latencies[latencies.size() * 0.99];

    FILE* fp = fopen(output_path, "a");
    fprintf(fp, "%f,%f,%f,%f,%f,%f\n", throughput, mean, p50, p90, p95, p99);
    fclose(fp);
}


