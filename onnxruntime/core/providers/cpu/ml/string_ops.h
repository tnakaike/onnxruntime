// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace ml {
class ConcatStrOp final : public OpKernel {
 public:
  explicit ConcatStrOp(const OpKernelInfo& info);
  common::Status Compute(OpKernelContext* context) const override;

 private:
  std::string separator_;
};
}  // namespace ml
}  // namespace onnxruntime
