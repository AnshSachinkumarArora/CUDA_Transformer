#pragma once

#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <stdio.h>
#include <cuda/cmath>

#define FULL_MASK 0xffffffff

static __forceinline__ __device__ float warp_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

static __forceinline__ __device__ float warp_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset)); 
    }
    return val;
}

template <const uint BLOCKSIZE, const uint NUMWARPS>
__global__ void naive_softmax(const float* x, float* out, int m, int n) {
    // --------------- SETUP --------------- //
    int row = blockIdx.y;
    int batch = blockIdx.z;

    //create required regs/smem
    __shared__ float smem[32];
    __shared__ float final_sum_inv;
    __shared__ float final_max;

    //per thread local values
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    //warp values
    int warpIdx = threadIdx.x % 32;
    int warpNum = threadIdx.x / 32;

    //move x/out pointer to correct row
    x += (batch * (m * n)) + (row * n);
    out += (batch * (m * n)) + (row * n);

    // --------------- MAX CALCULATION --------------- //    
    //no need for oob check since built into loop, no values over n can be read
    for (int i = threadIdx.x; i < n; i += BLOCKSIZE) {
        local_max = fmaxf(local_max, x[i]);
    }

    float partial_max = warp_max(local_max);

    //warp idx 0 copies the partials to smem
    if(warpIdx == 0) {
        smem[warpNum] = partial_max;
    }

    __syncthreads();

    //only warp 0 uses the partial smem values to calculate the final value
    if(warpNum == 0) {
        float max_val = (warpIdx < NUMWARPS) ? smem[warpIdx] : -INFINITY;
        float w0_max = warp_max(max_val);
        
        //warp 0 thread 0 writes to final value smem
        if(warpIdx == 0) {
            final_max = w0_max;
        }
    }

    __syncthreads();

    // --------------- SUM CALCULATION --------------- //
    for (int i = threadIdx.x; i < n; i += BLOCKSIZE) {
        local_sum += expf(x[i] - final_max);
    }

    float partial_sum = warp_sum(local_sum);
    
    if(warpIdx == 0) {
        smem[warpNum] = partial_sum;
    }

    __syncthreads();

    //only warp 0 uses the partial smem values to calculate the final value
    if(warpNum == 0) {
        float sum_val = (warpIdx < NUMWARPS) ? smem[warpIdx] : 0.0f;
        float w0_sum = warp_sum(sum_val);
        
        //warp 0 thread 0 writes to final value smem
        if(warpIdx == 0) {
            final_sum_inv = 1.0f/w0_sum;
        }
    }

    __syncthreads();

    // --------------- SOFTMAX CALCULATION --------------- //
    for (int i = threadIdx.x; i < n; i += BLOCKSIZE) {
        out[i] = (float)expf(x[i] - final_max)*final_sum_inv;
    }
}

template <const uint BLOCKSIZE, const uint NUMWARPS>
__global__ void naive_softmax_bwd(const float* probs, const float* dp, float* out, int m, int n) {
    // --------------- SETUP --------------- //
    int row = blockIdx.y;
    int batch = blockIdx.z;

    //create required regs/smem
    __shared__ float smem[32];
    __shared__ float final_sum;

    //per thread local values
    float local_sum = 0.0f;

    //warp values
    int warpIdx = threadIdx.x % 32;
    int warpNum = threadIdx.x / 32;

    //move x/out pointer to correct row
    probs += (batch * (m * n)) + (row * n);
    dp += (batch * (m * n)) + (row * n);
    out += (batch * (m * n)) + (row * n);

    //multiply all P with dP
    for(int col = threadIdx.x; col < n; col += BLOCKSIZE) {
        local_sum += probs[col] * dp[col];
    }

    //warp reduction for sum
    float partial_sum = warp_sum(local_sum);

    if(warpIdx == 0) {
        smem[warpNum] = partial_sum;
    }

    __syncthreads();

    if(warpNum == 0) {
        float sum_val = (warpIdx < NUMWARPS) ? smem[warpIdx] : 0.0f;
        float w0_sum = warp_sum(sum_val);
        
        //warp 0 thread 0 writes to final value smem
        if(warpIdx == 0) {
            final_sum = w0_sum;
        }
    }

    __syncthreads();

    //epilogue calculation
    for(int col = threadIdx.x; col < n; col += BLOCKSIZE) {
        out[col] = probs[col] * (dp[col] - final_sum);
    }
}