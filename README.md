# PyTorch C++ Module

PyTorch C++ Module package.

## Install

Before installing this repository, you need to install torch and CUDA
to build module.

```bash
pip install torch

# Source Install
git clone https://github.com/mjun0812/torch-cpp-module.git
cd torch-cpp-module
python setup.py build install --user

# pip install from github
pip install git+https://github.com/mjun0812/torch-cpp-module.git
```

## Provided Module

- [DCNv3](https://github.com/OpenGVLab/InternImage)
- [DCNv4](https://github.com/OpenGVLab/DCNv4)
- [DeformableAttention](https://github.com/fundamentalvision/Deformable-DETR)
