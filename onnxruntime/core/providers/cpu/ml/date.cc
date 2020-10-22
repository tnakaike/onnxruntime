// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/date.h"
#include <time.h>
/**
https://github.com/onnx/onnx/blob/master/onnx/defs/traditionalml/defs.cc
ONNX_ML_OPERATOR_SET_SCHEMA(
    Date,
    1,
    OpSchema()
        .SetDoc(Date_ver1_doc)
        .Input(0, "X", "Data to be processed.", "T1")
        .Output(0, "YM", "Months", "T2")
        .Output(1, "YD", "Days in a year", "T2")
        .Output(2, "MD", "Days in a month", "T2")
        .Output(3, "WD", "Days in a week", "T2")
        .Output(4, "H", "Hour", "T2")
        .Output(5, "M", "Minute", "T2")
        .TypeConstraint(
            "T1",
            {"tensor(string)"},
            "The input type must be a tensor of a string type.")
        .TypeConstraint(
            "T2",
            {"tensor(int32)"},
            "The type of each output is a tensor of a float type.")
        .Attr(
            "format",
            "Date format",
            AttributeProto::STRING,
            OPTIONAL_VALUE));
*/
using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_ML_KERNEL(
    Date,
    1,
    KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<std::string>())
                      .TypeConstraint("T2", DataTypeImpl::GetTensorType<int32_t>()),
    DateOp);

DateOp::DateOp(const OpKernelInfo& info) : OpKernel(info),
                                           format_(info.GetAttrOrDefault<std::string>("format", "")) {
}

common::Status DateOp::Compute(OpKernelContext* context) const {
  const auto* input_tensor_ptr = context->Input<Tensor>(0);
  ORT_ENFORCE(input_tensor_ptr != nullptr);

  const Tensor& X = *input_tensor_ptr;
  const TensorShape& x_shape = X.Shape();
  size_t x_size = x_shape.Size();
  const std::string *x_data = X.template Data<std::string>();

  Tensor* YM = context->Output(0, x_shape);
  Tensor* YD = context->Output(1, x_shape);
  Tensor* MD = context->Output(2, x_shape);
  Tensor* WD = context->Output(3, x_shape);
  Tensor* H = context->Output(4, x_shape);
  Tensor* M = context->Output(5, x_shape);
  int32_t *ym_data = YM->template MutableData<int32_t>();
  int32_t *yd_data = YD->template MutableData<int32_t>();
  int32_t *md_data = MD->template MutableData<int32_t>();
  int32_t *wd_data = WD->template MutableData<int32_t>();
  int32_t *h_data = H->template MutableData<int32_t>();
  int32_t *m_data = M->template MutableData<int32_t>();

  struct tm tm;
  for (size_t i = 0; i < x_size; i++) {
      std::string x_str = x_data[i];
      strptime(x_str.c_str(), format_.c_str(), &tm);
      ym_data[i] = tm.tm_mon + 1;
      yd_data[i] = tm.tm_yday + 1;
      md_data[i] = tm.tm_mday;
      wd_data[i] = tm.tm_wday + 1;
      h_data[i] = tm.tm_hour;
      m_data[i] = tm.tm_min;
      // printf("ym=%d yd=%d md=%d wd=%d h=%d m=%d\n", ym_data[i], yd_data[i], md_data[i], wd_data[i], h_data[i], m_data[i]);
  }
  
  return Status::OK();
}
}  // namespace ml
}  // namespace onnxruntime
