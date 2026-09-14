#pragma once

#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <stdio.h>
#include <cuda/cmath>

//STAGE 1: S = (Q@K^T)/sm_scale with causal masking. Outputs a matrix of shape: (B, T, T)
template <const int bm, const int bk, const int tm>
__global__ void causal_self_attention_stage_1(const float* qkv, float* out, int B, int T, int C, int nh, int hs, float sm_scale, bool causal) {
    assert(blockDim.x == bm * bk);

    int batch = blockIdx.z / nh;
    int head = blockIdx.z % nh;
    int outRow = blockIdx.y;
    int outCol = blockIdx.x;

    const float* startBatch = qkv + (batch * T * 3 * C);
    const float* q_ptr = startBatch + (outRow * bm * 3 * C) + (head * hs);
    const float* k_ptr = startBatch + (outCol * bm * 3 * C) + (head * hs) + C;
    out += (batch * nh * T * T) + (head * T * T) + (outRow * T * bm) + (outCol * bm);

    __shared__ float q[bm * bk];
    __shared__ float k[bm * bk];

    int token = threadIdx.x / bk;
    int feature = threadIdx.x % bk;
    int threadsPerRow = bm / tm;
    int threadOutRow = threadIdx.x / threadsPerRow;
    int threadOutCol = threadIdx.x % threadsPerRow;

    float acc[tm] = {0.0f};
    sm_scale = 1/sm_scale;

    for (int i = 0; i < hs; i += bk) {
        //absolute position of thread in q/k
        int qRow = outRow * bm + token;
        int kRow = outCol * bm + token;
        int col = feature + i;

        q[token * bk + feature] = (qRow < T && col < hs) ? q_ptr[(token * 3 * C) + feature + i] : 0.0f;
        k[token * bk + feature] = (kRow < T && col < hs) ? k_ptr[(token * 3 * C) + feature + i] : 0.0f;

        __syncthreads();

        for (int j = 0; j < bk; j++) {
            float temp = q[threadOutRow * bk + j];
            for (int l = 0; l < tm; l++) {
                acc[l] += k[(threadOutCol * tm * bk) + (l * bk) + j] * temp;
            }
        }
        
        __syncthreads();
    }

    __syncthreads();

    for(int i = 0; i < tm; i++) {
        int globalRow = (outRow * bm) + threadOutRow;
        int globalCol = (outCol * bm) + (threadOutCol * tm) + i;
        if(globalRow < T && globalCol < T) {
            //causal masking during final write
            out[(threadOutRow * T) + (threadOutCol * tm) + i] = (globalRow < globalCol && causal) ? -INFINITY : acc[i] * sm_scale;
        }
    }
}

//STAGE 3: O = P@V. Outputs a matrix of shape: (B, T, C)
template <const int bm, const int bk, const int tm>
__global__ void causal_self_attention_stage_3(const float* qkv, const float* p, float* out, int B, int T, int C, int nh, int hs) {
    assert(blockDim.x == bm * bk);

    int batch = blockIdx.z / nh;
    int head = blockIdx.z % nh;
    int outRow = blockIdx.y;
    int outCol = blockIdx.x;

    p += (batch * nh * T * T) + (head * T * T) + (outRow * bm * T);
    const float* v_ptr = qkv + (batch * T * 3 * C) + (head * hs) + (2 * C) + (outCol * bm);
    out += (batch * T * C) + (outRow * bm * C) + (head * hs) + (outCol * bm);

    __shared__ float probs[bm * bk];
    __shared__ float v[bk * bm];

    int rowP = threadIdx.x / bk;
    int colP = threadIdx.x % bk;
    int rowV = threadIdx.x / bm;
    int colV = threadIdx.x % bm;

    float acc[tm] = {0.0f};

    for (int i = 0; i < T; i += bk) {
        //absolute position of thread in v
        int rowVAbs = rowV + i;
        int colVAbs = outCol * bm + colV;
        //absolute position of thread in p
        int rowPAbs = outRow * bm + rowP;
        int colPAbs = colP + i;

        probs[(rowP * bk + colP)] = (rowPAbs < T && colPAbs < T) ? p[(rowP * T) + colP] : 0.0f;
        v[(rowV * bm + colV)] = (rowVAbs < T && colVAbs < hs) ? v_ptr[(rowV * 3 * C) + colV] : 0.0f;

        __syncthreads();

        p += bk;
        v_ptr += (3 * C * bk);

        for (int j = 0; j < bk; j++) {
            float temp = v[(j * bm + colV)];
            for (int k = 0; k < tm; k++) {
                acc[k] += probs[(rowV * bk * tm) + (bk * k) + j] * temp;
            }
        }

        __syncthreads();
    }

    __syncthreads();

    for (int i = 0; i < tm; i++) {
        int globalRow = (outRow * bm) + (rowV * tm) + i;
        int globalCol = (outCol * bm) + colV;
        if (globalRow < T && globalCol < hs) {
            out[(tm * C * rowV) + (i * C) + colV] = acc[i];
        }
    }
}