#!/bin/bash

# https://github.com/pytorch/pytorch/blob/591cb776af68e43f41cba1a29b6219341b75d951/torch/utils/cpp_extension.py#L1936
export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0+PTX"
python setup.py build install --user
