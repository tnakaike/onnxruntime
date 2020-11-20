// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/rangetransformer.h"
#include <cmath>
using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    RangeTransformer,
    1,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).MayInplace(0, 0),
    RangeTransformerOp<float>);

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    RangeTransformer,
    1,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()).MayInplace(0, 0),
    RangeTransformerOp<double>);

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    RangeTransformer,
    1,
    int64_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()).MayInplace(0, 0),
    RangeTransformerOp<int64_t>);

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    RangeTransformer,
    1,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()).MayInplace(0, 0),
    RangeTransformerOp<int32_t>);

template <typename T>
RangeTransformerOp<T>::RangeTransformerOp(const OpKernelInfo& info) : OpKernel(info),
                keys_upper_strings_(info.GetAttrsOrDefault<std::string>("keys_upper_strings")),
                keys_lower_strings_(info.GetAttrsOrDefault<std::string>("keys_lower_strings")),
                values_float_(info.GetAttrsOrDefault<float>("values_float")) {
    size_t keys_size = keys_upper_strings_.size();
    for (size_t i = 0; i < keys_size; i++){
        if ("None" == keys_upper_strings_[i]){
            keys_upper_float_.push_back(std::numeric_limits<float>::quiet_NaN());
        } else {
            keys_upper_float_.push_back(std::stof(keys_upper_strings_[i]));
        }
        if ("None" == keys_lower_strings_[i]){
            keys_lower_float_.push_back(std::numeric_limits<float>::quiet_NaN());
        } else {
            keys_lower_float_.push_back(std::stof(keys_lower_strings_[i]));
        }
    }
}

template <typename T>
common::Status RangeTransformerOp<T>::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);
  const TensorShape& x_shape = X.Shape();
  size_t x_size = x_shape.Size();
  const T* x_data = X.template Data<T>();

  size_t keys_size = keys_upper_strings_.size();
  if (keys_size != keys_lower_strings_.size()) {
    return Status(ONNXRUNTIME, FAIL, "keys_upper_strings and keys_lower_strings have different sizes.");
  }

  Tensor* Z = context->Output(0, x_shape);
  double *z_data = Z->template MutableData<double>();
  for (size_t i = 0; i < x_size; i++) {
      T x = x_data[i];
      z_data[i] = x;
      for (size_t j = 0; j < keys_size; j++){
          bool within_range = false;
          if (std::isnan(keys_upper_float_[j])){
              if (!std::isnan(keys_lower_float_[j])){
                  if (x >= keys_lower_float_[j]){
                      within_range = true;
                  }
              }
              // else skip
          } else if (x <= keys_upper_float_[j]){
              if (!std::isnan(keys_lower_float_[j])){
                  if (x >= keys_lower_float_[j]){
                      within_range = true;
                  }
              } else {
                  within_range = true;
              }
          }
          if (within_range){
              z_data[i] = values_float_[i];
              break;
          }
      }
  }

  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
