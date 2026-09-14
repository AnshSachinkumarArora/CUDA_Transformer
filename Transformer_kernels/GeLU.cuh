#pragma once

#include <cuda_runtime_api.h>
#include <cuda/cmath>

//foward pass
__global__ void gelu(const float* x, float* out, int numel) {
    //calculate absolute row/col positions
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    //bounds check for edge cases
    if (idx < numel) {
        float val = x[idx];
        out[idx] = val * normcdff(val);
    }
}

//backward pass
__global__ void gelu_bwd(const float* x, const float* dy, float* out, int numel) {
    //calculate absolute row/col positions
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < numel) {
        float val = x[idx];
        float grad = dy[idx];
        //pdf is the derivative of cdf
        float normpdff = 0.39894228f * expf(-0.5f * val * val);
        out[idx] = grad * (normcdff(val) + (val * normpdff));
    }
}