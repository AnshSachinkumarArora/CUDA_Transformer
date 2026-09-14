import os
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = 'cuda_transformer'

def get_extensions():

    extra_compile_args = {
        "cxx": [
            "-O3",
            "-fdiagnostics-color=always"
        ],
        "nvcc": [
            "-O3"
        ],
    }

    this_dir = os.path.dirname(os.path.curdir)
    sources = list(glob.glob(os.path.join(this_dir, "Transformer_launchers", "*.cu")))
    bindings = list(glob.glob(os.path.join(this_dir, "Transformer_launchers", "*.cpp")))
    sources += bindings

    ext_modules = [
        CUDAExtension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args
        )
    ]

    return ext_modules

setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="A transformer written in pure CUDA",
    cmdclass={"build_ext": BuildExtension}
)