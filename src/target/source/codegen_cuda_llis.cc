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
 * \file codegen_cuda_llis.cc
 */
#include "codegen_cuda_llis.h"

#include <cctype>
#include <iomanip>

#include "../../arith/pattern_match.h"

namespace tvm {
namespace codegen {

using namespace tir;

void CodeGenCUDALlis::AddFunction(const PrimFunc& f) {
  // clear previous generated state.
  this->InitFuncState(f);
  // reserve keywords
  ReserveKeywordsAsUnique();

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  CHECK(global_symbol.defined()) << "CodeGenCUDALlis: Expect PrimFunc to have the global_symbol attribute";
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);

  this->PrintFuncPrefix();
  this->stream << " " << static_cast<std::string>(global_symbol.value()) << "(";

  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0) stream << ", ";
    if (v.dtype().is_handle()) {
      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end()) {
        PrintStorageScope(it->second, stream);
      }

      PrintType(GetType(v), stream);
      // Register handle data type
      // TODO(tvm-team): consider simply keep type info in the
      // type annotation(via a normalizing rewriting).
      if (auto* ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto* prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }

      if (no_alias && restrict_keyword_.length() != 0) {
        stream << ' ' << restrict_keyword_;
      }
    } else {
      PrintType(GetType(v), stream);
    }
    stream << ' ' << vid;
  }

  this->PrintExtraParams();

  stream << ") {\n";

  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintFuncStart();
  this->PrintStmt(f->body);
  this->PrintFinalReturn();
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

void CodeGenCUDALlis::PrintExtraParams() {
  stream << ", llis::JobId __cuda_llis_job_id, llis::job::FinishedBlockNotifier* __cuda_llis_notifier";
}

void CodeGenCUDALlis::PrintFinalReturn() {
  stream << "__cuda_llis_exit: __cuda_llis_notifier->end(__cuda_llis_job_id);\n";
}

void CodeGenCUDALlis::PrintFuncStart() {
  stream << "__cuda_llis_notifier->start(__cuda_llis_job_id);\n";
}

std::string CodeGenCUDALlis::Finish() {
  // TODO(Kelvin): Replace every return instruction to jump to __cuda_llis_exit
  // I am not sure if this is necessary though. It looks like return is never generated...

#ifdef LLIS_MEASURE_BLOCK_TIME
  decl_stream << "#define LLIS_MEASURE_BLOCK_TIME\n";
#endif
#ifdef LLIS_FINISHED_BLOCK_NOTIFICATION_AGG
  decl_stream << "#define LLIS_FINISHED_BLOCK_NOTIFICATION_AGG\n";
#endif
  decl_stream << "#include <llis/job/finished_block_notifier.h>\n";

  return CodeGenCUDA::Finish();
}

}  // namespace codegen
}  // namespace tvm
