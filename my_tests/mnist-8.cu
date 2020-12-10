extern "C" __global__ void fused_nn_dense_add_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2, volatile unsigned* __cuda_kelvin_flag) {
  float T_dense_rf[1];
  __shared__ float red_buf0[64];
  __shared__ float T_dense[1];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 4; ++k_outer) {
    T_dense_rf[(0)] = (T_dense_rf[(0)] + (placeholder[(((k_outer * 64) + ((int)threadIdx.x)))] * placeholder1[((((((int)blockIdx.x) * 256) + (k_outer * 64)) + ((int)threadIdx.x)))]));
  }
  __syncthreads();
  ((volatile float*)red_buf0)[(((int)threadIdx.x))] = T_dense_rf[(0)];
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 32))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 16))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 8))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 4))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 2))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 1))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    T_dense[(0)] = ((volatile float*)red_buf0)[(0)];
  }
  if (((int)threadIdx.x) == 0) {
    T_add[(((int)blockIdx.x))] = (T_dense[(0)] + placeholder2[(((int)blockIdx.x))]);
  }
__cuda_kelvin_exit: if (threadIdx.x == 0 & __cuda_kelvin_flag != nullptr) atomicAdd((unsigned*)__cuda_kelvin_flag, 1);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, volatile unsigned* __cuda_kelvin_flag) {
  float compute[8];
  __shared__ float pad_temp_shared[224];
  __shared__ float placeholder_shared[128];
  #pragma unroll
  for (int ff_init = 0; ff_init < 4; ++ff_init) {
    #pragma unroll
    for (int yy_init = 0; yy_init < 2; ++yy_init) {
      compute[(((ff_init * 2) + yy_init))] = 0.000000e+00f;
    }
  }
  for (int ry_outer = 0; ry_outer < 5; ++ry_outer) {
    #pragma unroll
    for (int rx_outer = 0; rx_outer < 5; ++rx_outer) {
      __syncthreads();
      #pragma unroll
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
        pad_temp_shared[((((((int)threadIdx.z) * 56) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = (((((2 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 28) / 14)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 28) / 14)) + ry_outer) < 16)) && (2 <= (rx_outer + (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 14)))) && ((rx_outer + (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 14)) < 16)) ? placeholder[((((((((((int)threadIdx.z) * 392) + ((((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 28) * 196)) + (((int)blockIdx.y) * 28)) + (ry_outer * 14)) + rx_outer) + (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 28)) - 30))] : 0.000000e+00f);
      }
      #pragma unroll
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
        if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 3)) < 16) {
          if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 128) {
            if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 32) {
              placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((int)threadIdx.z) * 800) + (((int)threadIdx.x) * 75)) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 * 25)) + (ry_outer * 5)) + rx_outer))];
            }
          }
        }
      }
      __syncthreads();
      #pragma unroll
      for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
        #pragma unroll
        for (int ff = 0; ff < 4; ++ff) {
          #pragma unroll
          for (int yy = 0; yy < 2; ++yy) {
            compute[(((ff * 2) + yy))] = (compute[(((ff * 2) + yy))] + (pad_temp_shared[((((rc_inner * 28) + (yy * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 32) + (ff * 8)) + rc_inner))]));
          }
        }
      }
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 4; ++ax1_inner_inner_inner) {
    #pragma unroll
    for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 2; ++ax2_inner_inner_inner) {
      T_relu[((((((((int)threadIdx.z) * 784) + (ax1_inner_inner_inner * 196)) + (((int)blockIdx.y) * 28)) + (ax2_inner_inner_inner * 14)) + ((int)threadIdx.x)))] = max((compute[(((ax1_inner_inner_inner * 2) + ax2_inner_inner_inner))] + placeholder2[(((((int)threadIdx.z) * 4) + ax1_inner_inner_inner))]), 0.000000e+00f);
    }
  }
__cuda_kelvin_exit: if (threadIdx.x == 0 & __cuda_kelvin_flag != nullptr) atomicAdd((unsigned*)__cuda_kelvin_flag, 1);
}

extern "C" __global__ void fused_nn_max_pool2d_kernel0(float* __restrict__ placeholder, float* __restrict__ tensor, volatile unsigned* __cuda_kelvin_flag) {
  float tensor_local[1];
  tensor_local[(0)] = -3.402823e+38f;
  for (int rv = 0; rv < 3; ++rv) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      if (((int)threadIdx.x) < 256) {
        tensor_local[(0)] = max(tensor_local[(0)], placeholder[(((((((((int)threadIdx.x) >> 4) * 196) + (((((int)threadIdx.x) & 15) >> 2) * 42)) + (rv * 14)) + ((((int)threadIdx.x) & 3) * 3)) + rv1))]);
      }
    }
  }
  if (((int)threadIdx.x) < 256) {
    tensor[(((int)threadIdx.x))] = tensor_local[(0)];
  }
__cuda_kelvin_exit: if (threadIdx.x == 0 & __cuda_kelvin_flag != nullptr) atomicAdd((unsigned*)__cuda_kelvin_flag, 1);
}

extern "C" __global__ void fused_nn_max_pool2d_1_kernel0(float* __restrict__ placeholder, float* __restrict__ tensor, volatile unsigned* __cuda_kelvin_flag) {
  float tensor_local[1];
  tensor_local[(0)] = -3.402823e+38f;
  for (int rv = 0; rv < 2; ++rv) {
    for (int rv1 = 0; rv1 < 2; ++rv1) {
      if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 1568) {
        tensor_local[(0)] = max(tensor_local[(0)], placeholder[((((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 14) * 56) + (rv * 28)) + ((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 14) * 2)) + rv1))]);
      }
    }
  }
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 1568) {
    tensor[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = tensor_local[(0)];
  }
__cuda_kelvin_exit: if (threadIdx.x == 0 & __cuda_kelvin_flag != nullptr) atomicAdd((unsigned*)__cuda_kelvin_flag, 1);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, volatile unsigned* __cuda_kelvin_flag) {
  float compute[4];
  __shared__ float pad_temp_shared[28];
  __shared__ float placeholder_shared[8];
  compute[(0)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = (((2 <= ((int)blockIdx.y)) && (2 <= ((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) - 58))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[(((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = (((2 <= ((int)blockIdx.y)) && (1 <= ((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) - 57))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 1))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = ((2 <= ((int)blockIdx.y)) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) - 56))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 2))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = (((2 <= ((int)blockIdx.y)) && (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 27)) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) - 55))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 3))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = (((2 <= ((int)blockIdx.y)) && (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 26)) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) - 54))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 4))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = (((1 <= ((int)blockIdx.y)) && (2 <= ((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) - 30))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 5))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = (((1 <= ((int)blockIdx.y)) && (1 <= ((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) - 29))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 6))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = ((1 <= ((int)blockIdx.y)) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) - 28))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 7))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = (((1 <= ((int)blockIdx.y)) && (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 27)) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) - 27))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 8))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = (((1 <= ((int)blockIdx.y)) && (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 26)) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) - 26))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 9))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = ((2 <= ((((int)threadIdx.z) * 7) + ((int)threadIdx.x))) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) - 2))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 10))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = ((1 <= ((((int)threadIdx.z) * 7) + ((int)threadIdx.x))) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) - 1))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 11))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = placeholder[((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)))];
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 12))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = ((((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 27) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) + 1))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 13))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = ((((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 26) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) + 2))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 14))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = (((((int)blockIdx.y) < 27) && (2 <= ((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) + 26))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 15))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = (((((int)blockIdx.y) < 27) && (1 <= ((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) + 27))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 16))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = ((((int)blockIdx.y) < 27) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) + 28))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 17))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = (((((int)blockIdx.y) < 27) && (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 27)) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) + 29))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 18))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = (((((int)blockIdx.y) < 27) && (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 26)) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) + 30))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 19))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = (((((int)blockIdx.y) < 26) && (2 <= ((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) + 54))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 20))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = (((((int)blockIdx.y) < 26) && (1 <= ((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) + 55))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 21))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = ((((int)blockIdx.y) < 26) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) + 56))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 22))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = (((((int)blockIdx.y) < 26) && (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 27)) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) + 57))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 23))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = (((((int)blockIdx.y) < 26) && (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 26)) ? placeholder[(((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 7)) + ((int)threadIdx.x)) + 58))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) < 8) {
    if (((int)threadIdx.x) < 2) {
      placeholder_shared[(((((int)threadIdx.z) * 2) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 50) + (((int)threadIdx.x) * 25)) + 24))];
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  T_relu[((((((int)threadIdx.z) * 1568) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[((((int)threadIdx.z) * 2))]), 0.000000e+00f);
  T_relu[(((((((int)threadIdx.z) * 1568) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 14))] = max((compute[(2)] + placeholder2[((((int)threadIdx.z) * 2))]), 0.000000e+00f);
  T_relu[(((((((int)threadIdx.z) * 1568) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 784))] = max((compute[(1)] + placeholder2[(((((int)threadIdx.z) * 2) + 1))]), 0.000000e+00f);
  T_relu[(((((((int)threadIdx.z) * 1568) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 798))] = max((compute[(3)] + placeholder2[(((((int)threadIdx.z) * 2) + 1))]), 0.000000e+00f);
__cuda_kelvin_exit: if (threadIdx.x == 0 & __cuda_kelvin_flag != nullptr) atomicAdd((unsigned*)__cuda_kelvin_flag, 1);
}

extern "C" __global__ void fused_reshape_1_kernel0(float* __restrict__ T_reshape, float* __restrict__ placeholder, volatile unsigned* __cuda_kelvin_flag) {
  if (((int)threadIdx.x) < 256) {
    T_reshape[(((int)threadIdx.x))] = placeholder[(((int)threadIdx.x))];
  }
__cuda_kelvin_exit: if (threadIdx.x == 0 & __cuda_kelvin_flag != nullptr) atomicAdd((unsigned*)__cuda_kelvin_flag, 1);
}
