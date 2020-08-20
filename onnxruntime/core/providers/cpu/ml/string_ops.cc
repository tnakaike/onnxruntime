// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/string_ops.h"
#include <cmath>
/**
https://github.com/onnx/onnx/blob/master/onnx/defs/traditionalml/defs.cc
ONNX_ML_OPERATOR_SET_SCHEMA(
    ConcatStr,
    1,
    OpSchema()
        .SetDoc(ConcatStr_ver1_doc)
        .Input(0, "X", "Strings to be concatenated.", "T")
        .Input(1, "Y", "Strings to be concatenated.", "T")
        .Output(0, "Z", "Concatenated strings", "T")
        .TypeConstraint(
            "T",
            {"tensor(string)"},
            "The type of each input must be a tensor of a string type. The output type is a tensor of a string type.")
        .Attr(
            "separator",
            "String inserted between two input strings",
            AttributeProto::STRING,
            OPTIONAL_VALUE));
*/
using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_ML_KERNEL(
    ConcatStr,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<std::string>()),
    ConcatStrOp);

ConcatStrOp::ConcatStrOp(const OpKernelInfo& info) : OpKernel(info),
                                                     separator_(info.GetAttrOrDefault<std::string>("separator", "")) {
}

common::Status ConcatStrOp::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);
  const TensorShape& x_shape = X.Shape();
  size_t x_size = x_shape.Size();
  const std::string *x_data = X.template Data<std::string>();

  const Tensor& Y = *context->Input<Tensor>(1);
  const TensorShape& y_shape = Y.Shape();
  size_t y_size = y_shape.Size();
  const std::string *y_data = Y.template Data<std::string>();

  if (x_size != y_size) {
    return Status(ONNXRUNTIME, FAIL, "Two input tensors must have the same size");
  }

  Tensor* Z = context->Output(0, x_shape);
  std::string *z_data = Z->template MutableData<std::string>();
  for (size_t i = 0; i < x_size; i++) {
      std::string x_str = x_data[i];
      std::string y_str = y_data[i];
      std::string z_str = x_str + separator_ + y_str;
      printf("x=%s y=%s z=%s\n", x_str.c_str(), y_str.c_str(), z_str.c_str());
      z_data[i] = z_str;
  }

  return Status::OK();
}
}  // namespace ml
}  // namespace onnxruntime
