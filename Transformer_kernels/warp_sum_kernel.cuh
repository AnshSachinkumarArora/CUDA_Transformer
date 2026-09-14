#pragma once
#include <cuda_runtime_api.h>

#define FULL_MASK 0xffffffff

static __forceinline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

template <const uint BLOCKSIZE, const uint NUMWARPS>
__global__ void warp_sum_kernel(const float* x, float * out, int batch, int m, int n, 
    int stride_x_batch, int stride_x_row, int stride_x_col) {
    int col = blockIdx.x;

    x += (col * stride_x_col);
    out += col;

    //warp values
    int warpIdx = threadIdx.x % 32;
    int warpNum = threadIdx.x / 32;

    float local_sum = 0.0f;
    __shared__ float smem[NUMWARPS];

    //loop over all rows in fixed col and calculate per thread local sum
    for(int idx = threadIdx.x; idx < (batch * m); idx += BLOCKSIZE) {
        int b_idx = idx / m;
        int row = idx % m;
        local_sum += x[(b_idx * stride_x_batch) + (row * stride_x_row)];
    }

    float partial_sum = warp_reduce_sum(local_sum);
    
    if(warpIdx == 0) {
        smem[warpNum] = partial_sum;
    }

    __syncthreads();

    if(warpNum == 0) {
        float sum_val = (warpIdx < NUMWARPS) ? smem[warpIdx] : 0.0f;
        float w0_sum = warp_reduce_sum(sum_val);
        
        //warp 0 thread 0 writes final value to output
        if(warpIdx == 0) {
            out[0] = w0_sum;
        }
    }
}