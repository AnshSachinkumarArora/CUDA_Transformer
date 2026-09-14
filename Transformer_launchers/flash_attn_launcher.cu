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

#include "../Transformer_kernels/flash_attention.cuh"

namespace cuda_transformer {
    std::tuple<torch::Tensor, torch::Tensor> flash_attention_fwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal) {
        int B = Q.size(0);
        int nh = Q.size(1);
        int T = Q.size(2);
        int hs = Q.size(3);
        float sm_scale = std::sqrt(static_cast<float>(hs));

        TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
        TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
        TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");

        auto options = torch::TensorOptions().dtype(Q.dtype()).device(Q.device());
        torch::Tensor O = torch::empty({B, nh, T, hs}, options);
        torch::Tensor L = torch::empty({B, nh, T}, options);
        const float* q_ptr = Q.data_ptr<float>();
        const float* k_ptr = K.data_ptr<float>();
        const float* v_ptr = V.data_ptr<float>();
        float* o_ptr = O.data_ptr<float>();
        float* l_ptr = L.data_ptr<float>();

        constexpr int br = 16;
        constexpr int bc = 16;
        dim3 grid_dim(1, (T+br-1)/br, (B*nh));
        dim3 block_dim(br*bc);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        
        if (hs == 32) {
            flash_attention_fwd_kernel<br, bc, 32><<<grid_dim, block_dim, 0, stream>>>(q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, B, nh, T, sm_scale, causal,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), K.stride(0), K.stride(1), K.stride(2), K.stride(3), V.stride(0), V.stride(1), V.stride(2), V.stride(3));
        } else if (hs == 64) {
            flash_attention_fwd_kernel<br, bc, 64><<<grid_dim, block_dim, 0, stream>>>(q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, B, nh, T, sm_scale, causal,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), K.stride(0), K.stride(1), K.stride(2), K.stride(3), V.stride(0), V.stride(1), V.stride(2), V.stride(3));
        } else if (hs == 128) {
            flash_attention_fwd_kernel<br, bc, 128><<<grid_dim, block_dim, 0, stream>>>(q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, B, nh, T, sm_scale, causal,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), K.stride(0), K.stride(1), K.stride(2), K.stride(3), V.stride(0), V.stride(1), V.stride(2), V.stride(3));
        } else {
            TORCH_CHECK(false, "Unsupported head dimension! Only hs=32, hs=64, and hs=128 are supported.");
        }

        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "Kernel Failed: ", cudaGetErrorString(err));

        return {O, L};
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flash_attention_bwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, torch::Tensor dO, torch::Tensor L, bool causal) {
        int B = Q.size(0);
        int nh = Q.size(1);
        int T = Q.size(2);
        int hs = Q.size(3);
        float sm_scale = std::sqrt(static_cast<float>(hs));

        TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
        TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
        TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
        TORCH_CHECK(O.is_cuda(), "O must be a CUDA tensor");
        TORCH_CHECK(dO.is_cuda(), "dO must be a CUDA tensor");
        TORCH_CHECK(L.is_cuda(), "L must be a CUDA tensor");

        auto options = torch::TensorOptions().dtype(Q.dtype()).device(Q.device());
        torch::Tensor dQ = torch::empty({B, nh, T, hs}, options);
        torch::Tensor dK = torch::empty({B, nh, T, hs}, options);
        torch::Tensor dV = torch::empty({B, nh, T, hs}, options);
        torch::Tensor delta = torch::empty({B, nh, T}, options);
        const float* q_ptr = Q.data_ptr<float>();
        const float* k_ptr = K.data_ptr<float>();
        const float* v_ptr = V.data_ptr<float>();
        const float* o_ptr = O.data_ptr<float>();
        const float* dO_ptr = dO.data_ptr<float>();
        const float* l_ptr = L.data_ptr<float>();
        float* dQ_ptr = dQ.data_ptr<float>();
        float* dK_ptr = dK.data_ptr<float>();
        float* dV_ptr = dV.data_ptr<float>();
        float* delta_ptr = delta.data_ptr<float>();

        constexpr int br = 16;
        constexpr int bc = 16;
        dim3 block_dim(br*bc);
        dim3 block_dim_delta(1024);
        dim3 grid_dim_dq(1, (T+br-1)/br, (B*nh));
        dim3 grid_dim_dk_dv(1, (T+bc-1)/bc, (B*nh));
        dim3 grid_dim_delta(1, T, (B*nh));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        //get delta
        flash_attention_bwd_delta<<<grid_dim_delta, block_dim_delta, 0, stream>>>(o_ptr, dO_ptr, delta_ptr, B, nh, T, hs,
        O.stride(0), O.stride(1), O.stride(2), O.stride(3), dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3)); 

        if (hs == 32) {
            flash_attention_bwd_dq<br, bc, 32><<<grid_dim_dq, block_dim, 0, stream>>>(q_ptr, k_ptr, v_ptr, delta_ptr, l_ptr, dO_ptr, dQ_ptr, B, nh, T, sm_scale, causal,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), K.stride(0), K.stride(1), K.stride(2), K.stride(3), V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            L.stride(0), L.stride(1), L.stride(2), dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3));
            flash_attention_bwd_dk_dv<br, bc, 32><<<grid_dim_dk_dv, block_dim, 0, stream>>>(q_ptr, k_ptr, v_ptr, delta_ptr, l_ptr, dO_ptr, dK_ptr, dV_ptr, B, nh, T, sm_scale, causal,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), K.stride(0), K.stride(1), K.stride(2), K.stride(3), V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            L.stride(0), L.stride(1), L.stride(2), dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3));
        } else if (hs == 64) {
            flash_attention_bwd_dq<br, bc, 64><<<grid_dim_dq, block_dim, 0, stream>>>(q_ptr, k_ptr, v_ptr, delta_ptr, l_ptr, dO_ptr, dQ_ptr, B, nh, T, sm_scale, causal,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), K.stride(0), K.stride(1), K.stride(2), K.stride(3), V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            L.stride(0), L.stride(1), L.stride(2), dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3));
            flash_attention_bwd_dk_dv<br, bc, 64><<<grid_dim_dk_dv, block_dim, 0, stream>>>(q_ptr, k_ptr, v_ptr, delta_ptr, l_ptr, dO_ptr, dK_ptr, dV_ptr, B, nh, T, sm_scale, causal,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), K.stride(0), K.stride(1), K.stride(2), K.stride(3), V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            L.stride(0), L.stride(1), L.stride(2), dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3));
        } else if (hs == 128) {
            flash_attention_bwd_dq<br, bc, 128><<<grid_dim_dq, block_dim, 0, stream>>>(q_ptr, k_ptr, v_ptr, delta_ptr, l_ptr, dO_ptr, dQ_ptr, B, nh, T, sm_scale, causal,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), K.stride(0), K.stride(1), K.stride(2), K.stride(3), V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            L.stride(0), L.stride(1), L.stride(2), dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3));
            //need to scale down br/bc for hs=128 for dk/dv calculation since it exceeds gpu smem capacity
            //will replace with dynamic smem allocation to replace branching behaviour later
            dim3 grid_dim_dk_dv_128(1, (T+8-1)/8, (B*nh));
            flash_attention_bwd_dk_dv<8, 8, 128><<<grid_dim_dk_dv_128, block_dim, 0, stream>>>(q_ptr, k_ptr, v_ptr, delta_ptr, l_ptr, dO_ptr, dK_ptr, dV_ptr, B, nh, T, sm_scale, causal,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), K.stride(0), K.stride(1), K.stride(2), K.stride(3), V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            L.stride(0), L.stride(1), L.stride(2), dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3));
        } else {
            TORCH_CHECK(false, "Unsupported head dimension! Only hs=32, hs=64, and hs=128 are supported.");
        }

        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "Kernel Failed: ", cudaGetErrorString(err));

        return {dQ, dK, dV};
    }

    TORCH_LIBRARY_FRAGMENT(cuda_transformer, m) {
        m.def("flash_attention_fwd(Tensor Q, Tensor K, Tensor V, bool causal) -> (Tensor, Tensor)");
        m.def("flash_attention_bwd(Tensor Q, Tensor K, Tensor V, Tensor O, Tensor dO, Tensor L, bool causal) -> (Tensor, Tensor, Tensor)");
    }

    TORCH_LIBRARY_IMPL(cuda_transformer, CUDA, m) {
        m.impl("flash_attention_fwd", &flash_attention_fwd);
        m.impl("flash_attention_bwd", &flash_attention_bwd);
    }
}