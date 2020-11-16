// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace ml {
template <typename T>
class RangeTransformerOp final : public OpKernel {
 public:
  explicit RangeTransformerOp(const OpKernelInfo& info);
  common::Status Compute(OpKernelContext* context) const override;

 private:
  std::vector<std::string> keys_upper_strings_;
  std::vector<std::string> keys_lower_strings_;
  std::vector<float> keys_upper_float_;
  std::vector<float> keys_lower_float_;
  std::vector<float> values_float_;
};
}  // namespace ml
}  // namespace onnxruntime
