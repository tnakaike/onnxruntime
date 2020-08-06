// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace ml {
class ImputerOp final : public OpKernel {
 public:
  explicit ImputerOp(const OpKernelInfo& info);
  common::Status Compute(OpKernelContext* context) const override;

 private:
  std::vector<float> imputed_values_float_;
  float replaced_value_float_;
  std::vector<int64_t> imputed_values_int64_;
  int64_t replaced_value_int64_;
  std::vector<std::string> imputed_values_string_;
  std::string replaced_value_string_;
};

class SeqImputerOp final : public OpKernel {
 public:
  explicit SeqImputerOp(const OpKernelInfo& info);
  common::Status Compute(OpKernelContext* context) const override;

 private:
  common::Status Compute(OpKernelContext* context, const Tensor* X, Tensor* Y) const;

  std::vector<int64_t> input_positions_;
  std::vector<int64_t> output_positions_;
  std::vector<float> imputed_values_float_;
  float replaced_value_float_;
  std::vector<int64_t> imputed_values_int64_;
  int64_t replaced_value_int64_;
  std::vector<std::string> imputed_values_string_;
  std::string replaced_value_string_;
};
}  // namespace ml
}  // namespace onnxruntime
