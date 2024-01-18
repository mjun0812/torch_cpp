import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "torch_cpp/ops")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    # source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "**", "*.cu"))

    # sources = main_file + source_cpu
    sources = main_file
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    if CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            # https://discuss.pytorch.org/t/cuda-no-half2-operators-for-cuda-9-2/18365/3
            # https://github.com/pytorch/pytorch/blob/c3113514e91635e2cdb3fe26171a023897eb390d/cmake/Dependencies.cmake#L1642
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-O3",
        ]
    else:
        return None

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "torch_cpp._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


setup(
    name="torch_cpp",
    version="1.0.0",
    author="Junya Morioka",
    description="PyTorch C++ Module",
    install_requires=["torch"],
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
