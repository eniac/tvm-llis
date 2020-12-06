/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file codegen_cuda.h
 * \brief Utility to generate cuda code
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_CUDA_KELVIN_H_
#define TVM_TARGET_SOURCE_CODEGEN_CUDA_KELVIN_H_

#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <string>
#include <unordered_map>

#include "codegen_cuda.h"

namespace tvm {
namespace codegen {

class CodeGenCUDALlis : public CodeGenCUDA {
 public:
  void AddFunction(const PrimFunc& f) override;
  void PrintFinalReturn() override;  // NOLINT(*)
  virtual void PrintExtraParams();
  virtual void PrintFuncStart();
  std::string Finish();

  using CodeGenC::PrintType;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_CUDA_KELVIN_H_
