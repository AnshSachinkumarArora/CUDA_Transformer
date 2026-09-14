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

#include "../Transformer_kernels/softmax.cuh"

namespace cuda_transformer {
    torch::Tensor cuda_softmax(torch::Tensor x) {
        int batch = x.size(0);
        int m = x.size(1);
        int n = x.size(2);

        TORCH_CHECK(x.dim() == 3, "x must be a 3D tensor");
        TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
        TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

        auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
        torch::Tensor out = torch::empty({batch, m, n}, options);
        const float* x_ptr = x.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();

        const uint BLOCKSIZE = 1024;
        const uint NUMWARPS = BLOCKSIZE/32;
        dim3 grid_dim(1, m, batch);
        dim3 block_dim(BLOCKSIZE);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        naive_softmax<BLOCKSIZE, NUMWARPS><<<grid_dim, block_dim, 0, stream>>>(x_ptr, out_ptr, m, n);

        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "Kernel Failed: ", cudaGetErrorString(err));

        return out;
    }

    torch::Tensor cuda_softmax_bwd(torch::Tensor probs, torch::Tensor dp) {
        int batch = probs.size(0);
        int m = probs.size(1);
        int n = probs.size(2);

        TORCH_CHECK(probs.dim() == 3, "probs must be a 3D tensor");
        TORCH_CHECK(probs.is_cuda(), "probs must be a CUDA tensor");
        TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
        TORCH_CHECK(dp.dim() == 3, "dp must be a 3D tensor");
        TORCH_CHECK(dp.is_cuda(), "dp must be a CUDA tensor");
        TORCH_CHECK(dp.is_contiguous(), "dp must be contiguous");

        auto options = torch::TensorOptions().dtype(probs.dtype()).device(probs.device());
        torch::Tensor out = torch::empty({batch, m, n}, options);
        const float* probs_ptr = probs.data_ptr<float>();
        const float* dp_ptr = dp.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();

        const uint BLOCKSIZE = 1024;
        const uint NUMWARPS = BLOCKSIZE/32;
        dim3 grid_dim(1, m, batch);
        dim3 block_dim(BLOCKSIZE);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        naive_softmax_bwd<BLOCKSIZE, NUMWARPS><<<grid_dim, block_dim, 0, stream>>>(probs_ptr, dp_ptr, out_ptr, m, n);

        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "Kernel Failed: ", cudaGetErrorString(err));

        return out;
    }

    TORCH_LIBRARY_FRAGMENT(cuda_transformer, m) {
        m.def("cuda_softmax(Tensor x) -> Tensor");
        m.def("cuda_softmax_bwd(Tensor probs, Tensor dp) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(cuda_transformer, CUDA, m) {
        m.impl("cuda_softmax", &cuda_softmax);
        m.impl("cuda_softmax_bwd", &cuda_softmax_bwd);
    }
}