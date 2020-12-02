// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace ml {
class StringConcatOp final : public OpKernel {
 public:
  explicit StringConcatOp(const OpKernelInfo& info);
  common::Status Compute(OpKernelContext* context) const override;

 private:
  std::string separator_;
};

class StringSplitOp final : public OpKernel {
 public:
  explicit StringSplitOp(const OpKernelInfo& info);
  common::Status Compute(OpKernelContext* context) const override;

 private:
  std::string separator_;
  int64_t index_;
  int64_t keep_;
};

class StrLowerOp final : public OpKernel {
 public:
  explicit StrLowerOp(const OpKernelInfo& info);
  common::Status Compute(OpKernelContext* context) const override;
};
}  // namespace ml
}  // namespace onnxruntime
