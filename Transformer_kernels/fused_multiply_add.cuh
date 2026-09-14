#pragma once

#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <stdio.h>
#include <cuda/cmath>

// this is practically the same algorithm as 1d blocktiling sgemm but with bounds checking and bias addition in 1 kernel to implement linear layer
template <const int bm, const int bk, const int bn, const int tm>
__global__ void fused_multiply_add(const float* x, const float* weights, const float* bias, float* out, int m, int n, int k, 
    int stride_x_batch, int stride_x_row, int stride_x_col, int stride_weights_batch, int stride_weights_row, int stride_weights_col,
    int stride_bias_col) {
    assert(bm * bk == blockDim.x);
    assert(bn * bk == blockDim.x);

    int batch = blockIdx.z;
    int outCol = blockIdx.y;
    int outRow = blockIdx.x;

    //allocate smem buffer
    __shared__ float As[bm * bk];
    __shared__ float Bs[bk * bn];

    //coordinates for weights(bk * bn) and out(bm * bn) blocktiles
    const int row = threadIdx.x / bn;
    const int col = threadIdx.x % bn;

    //different coordinates for x since blocktile is bm * bk
    const int rowA = threadIdx.x / bk;
    const int colA = threadIdx.x % bk;

    //initial starting points for row in x, col in weights, and block in out
    x += (batch * stride_x_batch) + (bm * outRow * stride_x_row);
    weights += (batch * stride_weights_batch) + (bn * outCol * stride_weights_col);
    out += (batch * m * n) + ((bm * n) * outRow) + (bn * outCol);

    float acc[tm] = {0.0};

    for (int i = 0; i < k; i+=bk) {
        // absolute position of thread in matrix x
        int xRow = outRow * bm + rowA;
        int xCol = i + colA;
        // absolute position of thread in matrix weights
        int wRow = i + row;
        int wCol = outCol * bn + col;

        //here we use rowA since we are populating As and As has bk columns which means a new row starts after covering bk elements (read columns) per row of As
        As[rowA * bk + colA] = (xRow < m && xCol < k) ? x[rowA * stride_x_row + colA * stride_x_col] : 0.0f;
        //similar reasoning for Bs, Bs has bn columns thus reaching start of a row means covering row * bn elements
        Bs[row * bn + col] = (wRow < k && wCol < n) ? weights[row * stride_weights_row + col * stride_weights_col] : 0.0f;

        __syncthreads();

        x += bk * stride_x_col;
        weights += bk * stride_weights_row;

        for(int j = 0; j < bk; j++) {
            //since each thread is reading its own column from Bs which remains static, we read the element in the outer loop
            //this also facilitates algorithms where bk and tm don't align
            float temp = Bs[bn * j + col];
            for (int l = 0; l < tm; l++) {
                //here basically each thread covers a tm * 1 'block' of results in out, which means each thread is reading a 'block' 
                //of tm * bk elements from As (bm * bk) we just get the correct row the thread is responsible for populating in out
                //which gets us to the starting position of the correct tm * bk block in As. Once we are within the 'block' in As
                //we just get to the correct row within the block which ranges from 0 to (tm-1). Once within the correct row within
                //the correct 'block', we just add the corresponding column (j) which aligns with our loaded Bs value and perform our multiplication! 
                acc[l] += As[(tm * bk * row) + (l * bk) + j] * temp;
            } 
        }

        __syncthreads();
    }

    // single bias col since each thread owns a tm * 1 shape in the output 
    int biasCol = outCol * bn + col;
    float biasVal = (biasCol < n) ? bias[biasCol * stride_bias_col] : 0.0f;

    for (int i = 0; i < tm; i++) {
        //absolute position of thread in matrix out
        int outAbsRow = (outRow * bm) + (row * tm) + i;
        int outAbsCol = (outCol * bn) + col;
        //memory access only after bounds checking to avoid illegal memory access
        if (outAbsRow < m && outAbsCol < n) {
            //same logic as the inner loop above just applied to out. We just use n instead of bk since out shape is m * n.
            //each thread is now writing to a tm*1 block. To get to the correct 'block' the thread is responsible for, we need to 
            //cross tm * n * row elements, within the correct block, we need to get the appropriate row i (0 to (tm-1)), then to get
            //to the correct column within the remaining (n - (bn * outCol)) elements, we just add col and perform the sgemm computation 
            //add bias broadcasted over all tm  rows 
            out[(tm * n * row) + (i * n) + col] = acc[i] + biasVal;
        }
    }
}