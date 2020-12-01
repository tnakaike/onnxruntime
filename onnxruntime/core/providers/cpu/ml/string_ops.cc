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

ONNX_ML_OPERATOR_SET_SCHEMA(
    ConcatStr,
    1,
    OpSchema()
        .SetDoc(ConcatStr_ver1_doc)
        .Input(0, "X", "Strings to be concatenated.", "T")
        .Output(0, "Z", "Concatenated strings", "T")
        .TypeConstraint(
            "T",
            {"tensor(string)"},
            "The type of each input must be a tensor of a string type. The output type is a tensor of a string type.")
        .Attr(
            "separator",
            "String to split a string",
            AttributeProto::STRING,
            OPTIONAL_VALUE)
        .Attr(
            "index",
            "Index to split a string",
            AttributeProto::STRING,
            OPTIONAL_VALUE)
        .Attr(
            "keep",
            "Index for a split string",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          output_elem_type->set_elem_type(TensorProto::STRING);

          // Input and output shapes are the same.
          propagateShapeFromInputToOutput(ctx, 0, 0);
        }));
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
      // printf("x=%s y=%s z=%s\n", x_str.c_str(), y_str.c_str(), z_str.c_str());
      z_data[i] = z_str;
  }

  return Status::OK();
}

ONNX_CPU_OPERATOR_ML_KERNEL(
    SplitStr,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<std::string>()),
    SplitStrOp);

SplitStrOp::SplitStrOp(const OpKernelInfo& info) : OpKernel(info),
                                                   separator_(info.GetAttrOrDefault<std::string>("separator", "")),
                                                   index_(info.GetAttrOrDefault<int64_t>("index", -1)),
                                                   keep_(info.GetAttrOrDefault<int64_t>("keep", -1)) {
  if (index_ < -1)
    ORT_THROW("Expected index equal to or more than 0");
  else if (separator_.empty() && index_ < 0)
    ORT_THROW("Expected 'separator or index' attribute");
  else if (!separator_.empty() && index_ >= 0)
    ORT_THROW("Specify either separator or index");
}

common::Status SplitStrOp::Compute(OpKernelContext* context) const {
  if (!separator_.empty())
    return Status(ONNXRUNTIME, FAIL, "SpritStr with separator is not implemented yet.");
    
  const Tensor& X = *context->Input<Tensor>(0);
  const TensorShape& x_shape = X.Shape();
  size_t x_size = x_shape.Size();
  const std::string *x_data = X.template Data<std::string>();

  Tensor* Z = context->Output(0, x_shape);
  std::string *z_data = Z->template MutableData<std::string>();
  if (keep_ == -1) {
    for (size_t i = 0; i < x_size; i++) {
      std::string x_str = x_data[i];
      std::string z_str = x_str.substr(index_);
      // printf("x=%s z=%s\n", x_str.c_str(), z_str.c_str());
      z_data[i] = z_str;
    }
  } else {
    for (size_t i = 0; i < x_size; i++) {
      std::string x_str = x_data[i];
      std::string z_str = x_str.substr(0, index_);
      // printf("x=%s z=%s\n", x_str.c_str(), z_str.c_str());
      z_data[i] = z_str;
    }
  }

  return Status::OK();
}



/********************** StrLower **********************/


ONNX_CPU_OPERATOR_ML_KERNEL(
    StrLower,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<std::string>()),
    StrLowerOp);

StrLowerOp::StrLowerOp(const OpKernelInfo& info) : OpKernel(info){
}

common::Status StrLowerOp::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);
  const TensorShape& x_shape = X.Shape();
  size_t x_size = x_shape.Size();
  const std::string *x_data = X.template Data<std::string>();

  Tensor* Z = context->Output(0, x_shape);
  std::string *z_data = Z->template MutableData<std::string>();
  for (size_t i = 0; i < x_size; i++) {
      std::string x_str = x_data[i];
      std::string z_str;
      z_str.resize(x_size);
      std::transform(x_str.cbegin(), x_str.cend(), z_str.begin(), tolower);
      z_data[i] = z_str;
  }

  return Status::OK();
}
}  // namespace ml
}  // namespace onnxruntime
