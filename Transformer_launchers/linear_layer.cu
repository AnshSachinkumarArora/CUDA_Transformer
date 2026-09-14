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
#include <vector>

#include "../Transformer_kernels/fused_multiply_add.cuh"
#include "../Transformer_kernels/warp_sum_kernel.cuh"

namespace cuda_transformer{
    //x: (batch, m, k)
    //weights: (1, k, n), assumed that the matrix is pre-transposed
    //bias: (n,)
    //out: (batch, m, n)
    torch::Tensor cuda_fma(torch::Tensor x, torch::Tensor weights, torch::Tensor bias) {
        //in case weights is passed in as a 2d tensor
        if (weights.dim() == 2) {
            weights = weights.unsqueeze(0); 
        }

        int w_batch_stride = (weights.size(0) == 1) ? 0 : weights.stride(0);
        int x_batch = x.size(0);
        int m = x.size(1);
        int k = x.size(2);
        int n = weights.size(2);

        TORCH_CHECK(weights.size(1) == k, "weights.size(1) must match x.size(2)");
        TORCH_CHECK(bias.size(0) == n, "bias.size(0) must match weights.size(2)");
        TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
        TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

        auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
        torch::Tensor out = torch::empty({x_batch, m, n}, options);

        const float* x_ptr = x.data_ptr<float>();
        const float* weights_ptr = weights.data_ptr<float>();
        const float* bias_ptr = bias.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();

        const int bm = 64;
        const int bn = 64;
        const int bk = 8;
        const int tm = 8;
        
        dim3 grid_dim(((m + bm - 1)/bm), ((n + bn - 1)/bn), x_batch);
        dim3 block_dim((bm*bn)/tm);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        fused_multiply_add<bm, bk, bn, tm><<<grid_dim, block_dim, 0, stream>>>(x_ptr, weights_ptr, bias_ptr, out_ptr, m, n, k, 
        x.stride(0), x.stride(1), x.stride(2), w_batch_stride, weights.stride(1), weights.stride(2), bias.stride(0));

        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "Kernel Failed: ", cudaGetErrorString(err));

        return out;
    }

    //dy: (batch, m, n)
    //db: (n,)
    torch::Tensor db_calculation(torch::Tensor dy) {
        int batch = dy.size(0);
        int m = dy.size(1);
        int n = dy.size(2);

        auto options = torch::TensorOptions().dtype(dy.dtype()).device(dy.device());
        torch::Tensor db = torch::empty({n,}, options);

        const float* dy_ptr = dy.data_ptr<float>();
        float* db_ptr = db.data_ptr<float>();

        const uint BLOCKSIZE = 1024;
        const uint NUMWARPS = BLOCKSIZE/32;
        dim3 grid_dim(n);
        dim3 block_dim(BLOCKSIZE);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        warp_sum_kernel<BLOCKSIZE, NUMWARPS><<<grid_dim, block_dim, 0, stream>>>(dy_ptr, db_ptr, batch, m, n, dy.stride(0), dy.stride(1), dy.stride(2));

        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "Kernel Failed: ", cudaGetErrorString(err));

        return db;
    }

    //define cuda ops
    TORCH_LIBRARY_FRAGMENT(cuda_transformer, m) {
        m.def("cuda_fma(Tensor x, Tensor weights, Tensor bias) -> Tensor");
        m.def("db_calculation(Tensor dy) -> Tensor");
    }

    //register implementation
    TORCH_LIBRARY_IMPL(cuda_transformer, CUDA, m) {
        m.impl("cuda_fma", &cuda_fma);
        m.impl("db_calculation", &db_calculation);
    }
}