// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace ml {
class DateOp final : public OpKernel {
 public:
  explicit DateOp(const OpKernelInfo& info);
  common::Status Compute(OpKernelContext* context) const override;

 private:
  std::string format_;
};
}  // namespace ml
}  // namespace onnxruntime
