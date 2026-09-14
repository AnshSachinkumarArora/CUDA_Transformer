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

#include "../Transformer_kernels/causal_self_attention_fwd.cuh"
#include "../Transformer_kernels/softmax.cuh"
#include "../Transformer_kernels/batched_matrix_multiply.cuh"

namespace cuda_transformer {
    std::tuple<torch::Tensor, torch::Tensor> causal_self_attention_fwd(torch::Tensor qkv, int64_t nh, bool causal) {
        int B = qkv.size(0);
        int T = qkv.size(1);
        int C = qkv.size(2)/3;
        int hs = C/nh;
        float sm_scale = std::sqrt(static_cast<float>(hs));

        TORCH_CHECK(qkv.is_cuda(), "qkv must be a CUDA tensor");
        TORCH_CHECK(qkv.is_contiguous(), "qkv must be contiguous");

        auto options = torch::TensorOptions().dtype(qkv.dtype()).device(qkv.device());
        torch::Tensor s = torch::empty({B, nh, T, T}, options);
        const float* qkv_ptr = qkv.data_ptr<float>();
        float* s_ptr = s.data_ptr<float>();

        const int bm = 64;
        const int bk = 8;
        const int tm = 8;

        dim3 grid_dim_s1((T+bm-1)/bm, (T+bm-1)/bm, B*nh);
        dim3 block_dim_s1_s3((bm*bm)/tm);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        causal_self_attention_stage_1<bm, bk, tm><<<grid_dim_s1, block_dim_s1_s3, 0, stream>>>(qkv_ptr, s_ptr, B, T, C, nh, hs, sm_scale, causal);

        const uint BLOCKSIZE = 1024;
        const uint NUMWARPS = BLOCKSIZE/32;
        dim3 grid_dim_sm(1, T, B * nh);
        dim3 block_dim_sm(BLOCKSIZE);

        naive_softmax<BLOCKSIZE, NUMWARPS><<<grid_dim_sm, block_dim_sm, 0, stream>>>(s_ptr, s_ptr, T, T);

        torch::Tensor out = torch::empty({B, T, C}, options);
        float* out_ptr = out.data_ptr<float>();

        dim3 grid_dim_s3((hs+bm-1)/bm, (T+bm-1)/bm, B*nh);
        causal_self_attention_stage_3<bm, bk, tm><<<grid_dim_s3, block_dim_s1_s3, 0, stream>>>(qkv_ptr, s_ptr, out_ptr, B, T, C, nh, hs);

        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "Kernel Failed: ", cudaGetErrorString(err));

        return {out, s};
    }

    //input tensors are passed in as (B * nh, T, hs) except probs which is (B * nh, T, T)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> causal_self_attention_bwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor probs, torch::Tensor dO) {
        int B = Q.size(0); //the batch dims for all the tensors are actually batch * nh
        int T = Q.size(1);
        int hs = Q.size(2);
        
        TORCH_CHECK(Q.is_cuda(), "q must be a CUDA tensor");
        TORCH_CHECK(K.is_cuda(), "k must be a CUDA tensor");
        TORCH_CHECK(V.is_cuda(), "v must be a CUDA tensor");
        TORCH_CHECK(probs.is_cuda(), "probs must be a CUDA tensor");
        TORCH_CHECK(dO.is_cuda(), "dO must be a CUDA tensor");

        auto options = torch::TensorOptions().dtype(Q.dtype()).device(Q.device());
        torch::Tensor dq = torch::empty({B, T, hs}, options);
        torch::Tensor dk = torch::empty({B, T, hs}, options);
        torch::Tensor dv = torch::empty({B, T, hs}, options);
        torch::Tensor dp = torch::empty({B, T, T}, options);
        torch::Tensor ds = torch::empty({B, T, T}, options);

        const float* q_ptr = Q.data_ptr<float>();
        const float* k_ptr = K.data_ptr<float>();
        const float* v_ptr = V.data_ptr<float>();
        const float* probs_ptr = probs.data_ptr<float>();
        const float* dO_ptr = dO.data_ptr<float>();
        float* dq_ptr = dq.data_ptr<float>();
        float* dk_ptr = dk.data_ptr<float>();
        float* dv_ptr = dv.data_ptr<float>();
        float* dp_ptr = dp.data_ptr<float>();
        float* ds_ptr = ds.data_ptr<float>();

        //non softmax launch params
        const int bm = 64;
        const int bn = 64;
        const int bk = 8;
        const int tm = 8;
        dim3 block_dim((bm*bn)/tm);
        // Grid for dV, dQ, dK -> output is (T, hs)
        dim3 grid_dim_ths(((T + bm - 1)/bm), ((hs + bn - 1)/bn), B);
        // Grid for dP -> output is (T, T)
        dim3 grid_dim_tt(((T + bm - 1)/bm), ((T + bn - 1)/bn), B);
        
        //softmax launch params
        const uint BLOCKSIZE = 1024;
        const uint NUMWARPS = BLOCKSIZE/32;
        dim3 grid_dim_sm(1, T, B);
        dim3 block_dim_sm(BLOCKSIZE);

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        //dV
        cuda_bmm<bm, bk, bn, tm><<<grid_dim_ths, block_dim, 0, stream>>>(probs_ptr, dO_ptr, dv_ptr, T, T, hs, probs.stride(0), probs.stride(2), probs.stride(1), dO.stride(0), dO.stride(1), dO.stride(2));
        //dP
        cuda_bmm<bm, bk, bn, tm><<<grid_dim_tt, block_dim, 0, stream>>>(dO_ptr, v_ptr, dp_ptr, T, hs, T, dO.stride(0), dO.stride(1), dO.stride(2), V.stride(0), V.stride(2), V.stride(1));
        //dS
        naive_softmax_bwd<BLOCKSIZE, NUMWARPS><<<grid_dim_sm, block_dim_sm, 0, stream>>>(probs_ptr, dp_ptr, ds_ptr, T, T);
        //dQ (need to divide by sm_scale in python script)
        cuda_bmm<bm, bk, bn, tm><<<grid_dim_ths, block_dim, 0, stream>>>(ds_ptr, k_ptr, dq_ptr, T, T, hs, ds.stride(0), ds.stride(1), ds.stride(2), K.stride(0), K.stride(1), K.stride(2));
        //dK (need to divide by sm_scale in python script)
        cuda_bmm<bm, bk, bn, tm><<<grid_dim_ths, block_dim, 0, stream>>>(ds_ptr, q_ptr, dk_ptr, T, T, hs, ds.stride(0), ds.stride(2), ds.stride(1), Q.stride(0), Q.stride(1), Q.stride(2));

        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "Kernel Failed: ", cudaGetErrorString(err));

        return {dq, dk, dv};
    }

    TORCH_LIBRARY_FRAGMENT(cuda_transformer, m) {
        m.def("causal_self_attention_fwd(Tensor qkv, int nh, bool causal) -> (Tensor, Tensor)");
        m.def("causal_self_attention_bwd(Tensor Q, Tensor K, Tensor V, Tensor probs, Tensor dO) -> (Tensor, Tensor, Tensor)");
    }

    TORCH_LIBRARY_IMPL(cuda_transformer, CUDA, m) {
        m.impl("causal_self_attention_fwd", &causal_self_attention_fwd);
        m.impl("causal_self_attention_bwd", &causal_self_attention_bwd);
    }
}