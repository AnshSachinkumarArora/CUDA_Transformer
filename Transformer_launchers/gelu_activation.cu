#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <memory.h>
#include <cstdlib>
#include <cuda/cmath>
#include <iostream>

#include "../Transformer_kernels/GeLU.cuh"

namespace cuda_transformer {
    torch::Tensor gelu_activation(torch::Tensor x) {
        int batch = x.size(0);
        int m = x.size(1);
        int n = x.size(2);
        int numel = x.numel();

        TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
        TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

        auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
        torch::Tensor out = torch::empty({batch, m, n}, options);
        const float* x_ptr = x.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();

        int threads = 1024;
        dim3 grid_dim((numel+threads-1)/threads);
        dim3 block_dim(threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        gelu<<<grid_dim, block_dim, 0, stream>>>(x_ptr, out_ptr, numel);

        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "Kernel Failed: ", cudaGetErrorString(err));

        return out;
    }

    torch::Tensor gelu_activation_bwd(torch::Tensor x, torch::Tensor dy) {
        int batch = x.size(0);
        int m = x.size(1);
        int n = x.size(2);
        int numel = x.numel();

        TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
        TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
        TORCH_CHECK(dy.is_cuda(), "dy must be a CUDA tensor");
        TORCH_CHECK(dy.is_contiguous(), "dy must be contiguous");

        auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
        torch::Tensor out = torch::empty({batch, m, n}, options);
        const float* x_ptr = x.data_ptr<float>();
        const float* dy_ptr = dy.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();

        int threads = 1024;
        dim3 grid_dim((numel+threads-1)/threads);
        dim3 block_dim(threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        gelu_bwd<<<grid_dim, block_dim, 0, stream>>>(x_ptr, dy_ptr, out_ptr, numel);

        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "Kernel Failed: ", cudaGetErrorString(err));

        return out;
    }

    TORCH_LIBRARY_FRAGMENT(cuda_transformer, m) {
        m.def("gelu_activation(Tensor x) -> Tensor");
        m.def("gelu_activation_bwd(Tensor x, Tensor dy) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(cuda_transformer, CUDA, m) {
        m.impl("gelu_activation", &gelu_activation);
        m.impl("gelu_activation_bwd", &gelu_activation_bwd);
    }
}