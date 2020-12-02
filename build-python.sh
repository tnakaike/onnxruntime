#!/bin/bash

export ONNX_ML=1

# ./build.sh --config RelWithDebInfo --build_shared_lib --parallel --build_wheel
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --build_wheel --skip_tests

cd build/Linux/RelWithDebInfo/
python3 ../../../setup.py bdist_wheel
python3 -m pip install --upgrade dist/*.whl
