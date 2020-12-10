#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
void* __tvm_module_ctx = NULL;
static void* __tvm_set_device_packed = NULL;
static void* fused_nn_max_pool2d_kernel0_packed = NULL;
static void* fused_reshape_1_kernel0_packed = NULL;
static void* fused_nn_conv2d_add_nn_relu_kernel0_packed = NULL;
static void* fused_nn_max_pool2d_1_kernel0_packed = NULL;
static void* fused_nn_conv2d_add_nn_relu_1_kernel0_packed = NULL;
static void* fused_nn_dense_add_kernel0_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_max_pool2d(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  TVMValue stack[3];
  void* stack_tcode = stack;
  TVMValue stack1[5];
  void* stack_value = stack1;
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  void* tensor = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
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
  (((TVMValue*)stack_value)[0].v_handle) = placeholder;
  ((int32_t*)stack_tcode)[(0)] = 3;
  (((TVMValue*)stack_value)[1].v_handle) = tensor;
  ((int32_t*)stack_tcode)[(1)] = 3;
  (((TVMValue*)stack_value)[2].v_int64) = ((int64_t)1);
  ((int32_t*)stack_tcode)[(2)] = 0;
  (((TVMValue*)stack_value)[3].v_int64) = ((int64_t)1024);
  ((int32_t*)stack_tcode)[(3)] = 0;
  if (fused_nn_max_pool2d_kernel0_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_module_ctx, "fused_nn_max_pool2d_kernel0", &fused_nn_max_pool2d_kernel0_packed) != 0) {
      return -1;
    }
  }
  TVMValue ret_val1;
  int ret_type_code1;
  if (TVMFuncCall(fused_nn_max_pool2d_kernel0_packed, (TVMValue*) stack_value, (int*) stack_tcode, 4, &ret_val1, &ret_type_code1) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_reshape_1(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  TVMValue stack[3];
  void* stack_tcode = stack;
  TVMValue stack1[5];
  void* stack_value = stack1;
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  void* T_reshape = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
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
  (((TVMValue*)stack_value)[0].v_handle) = T_reshape;
  ((int32_t*)stack_tcode)[(0)] = 3;
  (((TVMValue*)stack_value)[1].v_handle) = placeholder;
  ((int32_t*)stack_tcode)[(1)] = 3;
  (((TVMValue*)stack_value)[2].v_int64) = ((int64_t)1);
  ((int32_t*)stack_tcode)[(2)] = 0;
  (((TVMValue*)stack_value)[3].v_int64) = ((int64_t)1024);
  ((int32_t*)stack_tcode)[(3)] = 0;
  if (fused_reshape_1_kernel0_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_module_ctx, "fused_reshape_1_kernel0", &fused_reshape_1_kernel0_packed) != 0) {
      return -1;
    }
  }
  TVMValue ret_val1;
  int ret_type_code1;
  if (TVMFuncCall(fused_reshape_1_kernel0_packed, (TVMValue*) stack_value, (int*) stack_tcode, 4, &ret_val1, &ret_type_code1) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_nn_relu(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  TVMValue stack[9];
  void* stack_tcode = stack;
  TVMValue stack1[17];
  void* stack_value = stack1;
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = ((int32_t*)arg_type_ids)[(3)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* placeholder2 = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  void* T_relu = (((DLTensor*)arg3)[0].data);
  void* arg3_shape = (((DLTensor*)arg3)[0].shape);
  void* arg3_strides = (((DLTensor*)arg3)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
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
  (((TVMValue*)stack_value)[0].v_handle) = placeholder;
  ((int32_t*)stack_tcode)[(0)] = 3;
  (((TVMValue*)stack_value)[1].v_handle) = placeholder1;
  ((int32_t*)stack_tcode)[(1)] = 3;
  (((TVMValue*)stack_value)[2].v_handle) = T_relu;
  ((int32_t*)stack_tcode)[(2)] = 3;
  (((TVMValue*)stack_value)[3].v_handle) = placeholder2;
  ((int32_t*)stack_tcode)[(3)] = 3;
  (((TVMValue*)stack_value)[4].v_int64) = ((int64_t)1);
  ((int32_t*)stack_tcode)[(4)] = 0;
  (((TVMValue*)stack_value)[5].v_int64) = ((int64_t)7);
  ((int32_t*)stack_tcode)[(5)] = 0;
  (((TVMValue*)stack_value)[6].v_int64) = ((int64_t)1);
  ((int32_t*)stack_tcode)[(6)] = 0;
  (((TVMValue*)stack_value)[7].v_int64) = ((int64_t)4);
  ((int32_t*)stack_tcode)[(7)] = 0;
  (((TVMValue*)stack_value)[8].v_int64) = ((int64_t)1);
  ((int32_t*)stack_tcode)[(8)] = 0;
  (((TVMValue*)stack_value)[9].v_int64) = ((int64_t)14);
  ((int32_t*)stack_tcode)[(9)] = 0;
  (((TVMValue*)stack_value)[10].v_int64) = ((int64_t)4);
  ((int32_t*)stack_tcode)[(10)] = 0;
  (((TVMValue*)stack_value)[11].v_int64) = ((int64_t)1);
  ((int32_t*)stack_tcode)[(11)] = 0;
  (((TVMValue*)stack_value)[12].v_int64) = ((int64_t)14);
  ((int32_t*)stack_tcode)[(12)] = 0;
  (((TVMValue*)stack_value)[13].v_int64) = ((int64_t)4);
  ((int32_t*)stack_tcode)[(13)] = 0;
  (((TVMValue*)stack_value)[14].v_int64) = ((int64_t)1);
  ((int32_t*)stack_tcode)[(14)] = 0;
  (((TVMValue*)stack_value)[15].v_int64) = ((int64_t)14);
  ((int32_t*)stack_tcode)[(15)] = 0;
  if (fused_nn_conv2d_add_nn_relu_kernel0_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_module_ctx, "fused_nn_conv2d_add_nn_relu_kernel0", &fused_nn_conv2d_add_nn_relu_kernel0_packed) != 0) {
      return -1;
    }
  }
  TVMValue ret_val1;
  int ret_type_code1;
  if (TVMFuncCall(fused_nn_conv2d_add_nn_relu_kernel0_packed, (TVMValue*) stack_value, (int*) stack_tcode, 16, &ret_val1, &ret_type_code1) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_max_pool2d_1(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  TVMValue stack[3];
  void* stack_tcode = stack;
  TVMValue stack1[5];
  void* stack_value = stack1;
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  void* tensor = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
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
  (((TVMValue*)stack_value)[0].v_handle) = placeholder;
  ((int32_t*)stack_tcode)[(0)] = 3;
  (((TVMValue*)stack_value)[1].v_handle) = tensor;
  ((int32_t*)stack_tcode)[(1)] = 3;
  (((TVMValue*)stack_value)[2].v_int64) = ((int64_t)2);
  ((int32_t*)stack_tcode)[(2)] = 0;
  (((TVMValue*)stack_value)[3].v_int64) = ((int64_t)1024);
  ((int32_t*)stack_tcode)[(3)] = 0;
  if (fused_nn_max_pool2d_1_kernel0_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_module_ctx, "fused_nn_max_pool2d_1_kernel0", &fused_nn_max_pool2d_1_kernel0_packed) != 0) {
      return -1;
    }
  }
  TVMValue ret_val1;
  int ret_type_code1;
  if (TVMFuncCall(fused_nn_max_pool2d_1_kernel0_packed, (TVMValue*) stack_value, (int*) stack_tcode, 4, &ret_val1, &ret_type_code1) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_nn_relu_1(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  TVMValue stack[9];
  void* stack_tcode = stack;
  TVMValue stack1[17];
  void* stack_value = stack1;
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = ((int32_t*)arg_type_ids)[(3)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* placeholder2 = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  void* T_relu = (((DLTensor*)arg3)[0].data);
  void* arg3_shape = (((DLTensor*)arg3)[0].shape);
  void* arg3_strides = (((DLTensor*)arg3)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
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
  (((TVMValue*)stack_value)[0].v_handle) = placeholder;
  ((int32_t*)stack_tcode)[(0)] = 3;
  (((TVMValue*)stack_value)[1].v_handle) = placeholder1;
  ((int32_t*)stack_tcode)[(1)] = 3;
  (((TVMValue*)stack_value)[2].v_handle) = T_relu;
  ((int32_t*)stack_tcode)[(2)] = 3;
  (((TVMValue*)stack_value)[3].v_handle) = placeholder2;
  ((int32_t*)stack_tcode)[(3)] = 3;
  (((TVMValue*)stack_value)[4].v_int64) = ((int64_t)1);
  ((int32_t*)stack_tcode)[(4)] = 0;
  (((TVMValue*)stack_value)[5].v_int64) = ((int64_t)28);
  ((int32_t*)stack_tcode)[(5)] = 0;
  (((TVMValue*)stack_value)[6].v_int64) = ((int64_t)1);
  ((int32_t*)stack_tcode)[(6)] = 0;
  (((TVMValue*)stack_value)[7].v_int64) = ((int64_t)4);
  ((int32_t*)stack_tcode)[(7)] = 0;
  (((TVMValue*)stack_value)[8].v_int64) = ((int64_t)1);
  ((int32_t*)stack_tcode)[(8)] = 0;
  (((TVMValue*)stack_value)[9].v_int64) = ((int64_t)14);
  ((int32_t*)stack_tcode)[(9)] = 0;
  (((TVMValue*)stack_value)[10].v_int64) = ((int64_t)4);
  ((int32_t*)stack_tcode)[(10)] = 0;
  (((TVMValue*)stack_value)[11].v_int64) = ((int64_t)1);
  ((int32_t*)stack_tcode)[(11)] = 0;
  (((TVMValue*)stack_value)[12].v_int64) = ((int64_t)14);
  ((int32_t*)stack_tcode)[(12)] = 0;
  (((TVMValue*)stack_value)[13].v_int64) = ((int64_t)4);
  ((int32_t*)stack_tcode)[(13)] = 0;
  (((TVMValue*)stack_value)[14].v_int64) = ((int64_t)1);
  ((int32_t*)stack_tcode)[(14)] = 0;
  (((TVMValue*)stack_value)[15].v_int64) = ((int64_t)14);
  ((int32_t*)stack_tcode)[(15)] = 0;
  if (fused_nn_conv2d_add_nn_relu_1_kernel0_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_module_ctx, "fused_nn_conv2d_add_nn_relu_1_kernel0", &fused_nn_conv2d_add_nn_relu_1_kernel0_packed) != 0) {
      return -1;
    }
  }
  TVMValue ret_val1;
  int ret_type_code1;
  if (TVMFuncCall(fused_nn_conv2d_add_nn_relu_1_kernel0_packed, (TVMValue*) stack_value, (int*) stack_tcode, 16, &ret_val1, &ret_type_code1) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_dense_add(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  TVMValue stack[4];
  void* stack_tcode = stack;
  TVMValue stack1[8];
  void* stack_value = stack1;
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = ((int32_t*)arg_type_ids)[(3)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* placeholder2 = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  void* T_add = (((DLTensor*)arg3)[0].data);
  void* arg3_shape = (((DLTensor*)arg3)[0].shape);
  void* arg3_strides = (((DLTensor*)arg3)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
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
  (((TVMValue*)stack_value)[0].v_handle) = placeholder;
  ((int32_t*)stack_tcode)[(0)] = 3;
  (((TVMValue*)stack_value)[1].v_handle) = placeholder1;
  ((int32_t*)stack_tcode)[(1)] = 3;
  (((TVMValue*)stack_value)[2].v_handle) = T_add;
  ((int32_t*)stack_tcode)[(2)] = 3;
  (((TVMValue*)stack_value)[3].v_handle) = placeholder2;
  ((int32_t*)stack_tcode)[(3)] = 3;
  (((TVMValue*)stack_value)[4].v_int64) = ((int64_t)1);
  ((int32_t*)stack_tcode)[(4)] = 0;
  (((TVMValue*)stack_value)[5].v_int64) = ((int64_t)10);
  ((int32_t*)stack_tcode)[(5)] = 0;
  (((TVMValue*)stack_value)[6].v_int64) = ((int64_t)64);
  ((int32_t*)stack_tcode)[(6)] = 0;
  if (fused_nn_dense_add_kernel0_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_module_ctx, "fused_nn_dense_add_kernel0", &fused_nn_dense_add_kernel0_packed) != 0) {
      return -1;
    }
  }
  TVMValue ret_val1;
  int ret_type_code1;
  if (TVMFuncCall(fused_nn_dense_add_kernel0_packed, (TVMValue*) stack_value, (int*) stack_tcode, 7, &ret_val1, &ret_type_code1) != 0) {
    return -1;
  }
  return 0;
}
