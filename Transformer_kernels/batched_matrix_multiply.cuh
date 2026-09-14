#pragma once

#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <stdio.h>
#include <cuda/cmath>

template <int bm, int bk, int bn, int tm>
__global__ void cuda_bmm(const float* A, const float* B, float* out, int m, int k, int n, 
    int stride_a_batch, int stride_a_row, int stride_a_col, int stride_b_batch, int stride_b_row, int stride_b_col) {
    assert(bm * bk == blockDim.x);
    assert(bn * bk == blockDim.x);

    //global pointer setup
    int batch = blockIdx.z;
    int outCol = blockIdx.y;
    int outRow = blockIdx.x;

    //allocate smem buffer
    __shared__ float As[bm * bk];
    __shared__ float Bs[bk * bn];

    //coordinates for B(bk * bn) and out(bm * bn) blocktiles
    const int row = threadIdx.x / bn;
    const int col = threadIdx.x % bn;

    //different coordinates for A since blocktile is bm * bk
    const int rowA = threadIdx.x / bk;
    const int colA = threadIdx.x % bk;

    //move pointers to appropriate starting positions
    A += (batch * stride_a_batch) + (stride_a_row * bm * outRow);
    B += (batch * stride_b_batch) + (stride_b_col * bn * outCol);
    out += (batch * (m * n)) + (bm * n * outRow) + (bn * outCol);

    float acc[tm] = {0.0};

    for(int i = 0; i < k; i += bk) {
        // absolute position of thread in matrix A
        int aRow = outRow * bm + rowA;
        int aCol = i + colA;
        // absolute position of thread in matrix B
        int bRow = i + row;
        int bCol = outCol * bn + col;

        //here we use rowA since we are populating As and As has bk columns which means a new row starts after covering bk elements (read columns) per row of As
        As[rowA * bk + colA] = (aRow < m && aCol < k) ? A[rowA * stride_a_row + colA * stride_a_col] : 0.0f;
        //similar reasoning for Bs, Bs has bn columns thus reaching start of a row means covering row * bn elements
        Bs[row * bn + col] = (bRow < k && bCol < n) ? B[row * stride_b_row + col * stride_b_col] : 0.0f;

        __syncthreads();

        A += bk * stride_a_col;
        B += bk * stride_b_row;

        for(int j = 0; j < bk; j++) {
            float temp = Bs[bn * j + col];
            for (int l = 0; l < tm; l++) {
                acc[l] += As[(tm * bk * row) + (l * bk) + j] * temp;
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < tm; i++) {
        //absolute position of thread in matrix out
        int outAbsRow = (outRow * bm) + (row * tm) + i;
        int outAbsCol = (outCol * bn) + col;
        //memory access only after bounds checking to avoid illegal memory access
        if (outAbsRow < m && outAbsCol < n) {
            out[(tm * n * row) + (i * n) + col] = acc[i];
        }
    }
}