#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
void* __tvm_module_ctx = NULL;
static void* __tvm_set_device_packed = NULL;
static void* myadd_kernel0_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t myadd(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  TVMValue stack[5];
  void* stack_tcode = stack;
  TVMValue stack1[10];
  void* stack_value = stack1;
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* A = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  int32_t n = ((int32_t)((int64_t*)arg0_shape)[(0)]);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t stride = ((n == 1) ? 0 : ((arg0_strides == NULL) ? 1 : ((int32_t)((int64_t*)arg0_strides)[(0)])));
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  void* B = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  int32_t stride1 = ((n == 1) ? 0 : ((arg1_strides == NULL) ? 1 : ((int32_t)((int64_t*)arg1_strides)[(0)])));
  void* C = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  int32_t stride2 = ((n == 1) ? 0 : ((arg2_strides == NULL) ? 1 : ((int32_t)((int64_t*)arg2_strides)[(0)])));
  (((TVMValue*)stack_value)[0].v_int64) = ((int64_t)2);
  ((int32_t*)stack_tcode)[(0)] = 0;
  (((TVMValue*)stack_value)[1].v_int64) = ((int64_t)dev_id);
  ((int32_t*)stack_tcode)[(1)] = 0;
  if (__tvm_set_device_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_module_ctx, "__tvm_set_device", &__tvm_set_device_packed) != 0) {
      return -1;
    }
  }
  TVMValue ret_val;
  int ret_type_code;
  if (TVMFuncCall(__tvm_set_device_packed, (TVMValue*) stack_value, (int*) stack_tcode, 2, &ret_val, &ret_type_code) != 0) {
    return -1;
  }
  (((TVMValue*)stack_value)[0].v_handle) = C;
  ((int32_t*)stack_tcode)[(0)] = 3;
  (((TVMValue*)stack_value)[1].v_handle) = A;
  ((int32_t*)stack_tcode)[(1)] = 3;
  (((TVMValue*)stack_value)[2].v_handle) = B;
  ((int32_t*)stack_tcode)[(2)] = 3;
  (((TVMValue*)stack_value)[3].v_int64) = ((int64_t)n);
  ((int32_t*)stack_tcode)[(3)] = 0;
  (((TVMValue*)stack_value)[4].v_int64) = ((int64_t)stride);
  ((int32_t*)stack_tcode)[(4)] = 0;
  (((TVMValue*)stack_value)[5].v_int64) = ((int64_t)stride1);
  ((int32_t*)stack_tcode)[(5)] = 0;
  (((TVMValue*)stack_value)[6].v_int64) = ((int64_t)stride2);
  ((int32_t*)stack_tcode)[(6)] = 0;
  (((TVMValue*)stack_value)[7].v_int64) = ((int64_t)((n + 63) >> 6));
  ((int32_t*)stack_tcode)[(7)] = 0;
  (((TVMValue*)stack_value)[8].v_int64) = ((int64_t)64);
  ((int32_t*)stack_tcode)[(8)] = 0;
  if (myadd_kernel0_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_module_ctx, "myadd_kernel0", &myadd_kernel0_packed) != 0) {
      return -1;
    }
  }
  TVMValue ret_val1;
  int ret_type_code1;
  if (TVMFuncCall(myadd_kernel0_packed, (TVMValue*) stack_value, (int*) stack_tcode, 9, &ret_val1, &ret_type_code1) != 0) {
    return -1;
  }
  return 0;
}


extern "C" __global__ void myadd_kernel0(float* __restrict__ C, float* __restrict__ A, float* __restrict__ B, int n, int stride, int stride1, int stride2) {
  if (((int)blockIdx.x) < (n >> 6)) {
    C[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride2))] = (A[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride))] + B[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride1))]);
  } else {
    if (((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) < n) {
      C[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride2))] = (A[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride))] + B[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride1))]);
    }
  }
}

int main() {
}

