#pragma once

#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>

class CUDAKelvinBench {
  public:
    CUDAKelvinBench(unsigned num_streams) {
        num_streams_ = num_streams;
        cudaMallocHost(&flags_, num_streams_ * sizeof(*flags_));
        streams_.resize(num_streams_);
        for (unsigned i = 0; i < num_streams_; ++i) {
            cuStreamCreate(&(streams_[i]), CU_STREAM_NON_BLOCKING);
        }
    }

    static CUDAKelvinBench* get(unsigned num_streams = -1) {
        static CUDAKelvinBench* tvm_bench = nullptr;
        if (tvm_bench == nullptr && num_streams > 0) {
            tvm_bench = new CUDAKelvinBench(num_streams);
        }

        return tvm_bench;
    }

    volatile unsigned* get_flags() {
        return flags_;
    }

    std::vector<CUstream>* get_streams() {
        return &streams_;
    }

    std::pair<volatile unsigned*, CUstream> acquire() {
        volatile unsigned* flag = flags_ + next_;
        CUstream stream = streams_[next_];
        ++next_;
        return std::make_pair(flag, stream);
    }

  private:
    unsigned num_streams_;
    volatile unsigned* flags_;
    std::vector<CUstream> streams_;
    unsigned next_ = 0;
};

