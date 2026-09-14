#pragma once

#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <stdio.h>
#include <cuda/cmath>

#define FULL_MASK 0xffffffff

//helper functions for row max/sum
static __forceinline__ __device__ float row_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset)); 
    }
    return __shfl_sync(FULL_MASK, val, 0);
}

static __forceinline__ __device__ float row_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return __shfl_sync(FULL_MASK, val, 0);
}

// ============= FORWARD PASS =============
template <int br, int bc, int hs>
__global__ void flash_attention_fwd_kernel(const float* Q, const float* K, const float* V, float* O, float* L, int B, int nh, int T, float sm_scale, bool causal,
    int stride_q_b, int stride_q_nh, int stride_q_t, int stride_q_hs,
    int stride_k_b, int stride_k_nh, int stride_k_t, int stride_k_hs,
    int stride_v_b, int stride_v_nh, int stride_v_t, int stride_v_hs) {
    //get to correct batch and head
    int batch = blockIdx.z / nh;
    int head = blockIdx.z % nh;
    //get the correct br width "row" being operated on
    //by the thread block on the output. outCol is unused
    //as we're calculating all cols per outRow per block
    int outRow = blockIdx.y;
    //warp setup for warp level primitives
    //each warp processes 1 row in Br
    int num_warps = blockDim.x / 32;
    int warpIdx = threadIdx.x / 32;
    int laneIdx = threadIdx.x % 32;
    //setup sm_scale
    sm_scale = 1/sm_scale;

    //smem setup 
    __shared__ float q[br * hs];
    __shared__ float k[bc * hs];
    __shared__ float v[bc * hs];
    __shared__ float s[br * bc];
    __shared__ float o[br * hs];
    __shared__ float m[br]; 
    __shared__ float l[br]; 

    for(int i = threadIdx.x; i < br; i += blockDim.x) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
    }

    __syncthreads();

    //starting pointers
    Q += (batch * stride_q_b) + (head * stride_q_nh) + (outRow * br * stride_q_t);
    K += (batch * stride_k_b) + (head * stride_k_nh);
    V += (batch * stride_v_b) + (head * stride_v_nh);
    O += (batch * nh * T * hs) + (head * T * hs) + (outRow * br * hs);
    L += (batch * nh * T) + (head * T) + (outRow * br);

    //copy q into smem and also initialize o
    for(int i = threadIdx.x; i < (br*hs); i += blockDim.x) {
        int row = i / hs;
        int col = i % hs;
        int absQRow = (outRow * br + row);
        q[i] = (absQRow < T) ? Q[(row * stride_q_t) + (col * stride_q_hs)] : 0.0f;
        o[i] = 0.0f;
    }

    //stage setup for causal masking Q
    int qEnd = causal ? min((outRow + 1) * br, T) : T;

    __syncthreads();

    //online softmax computation
    //loop over the entire K/V matrices
    for(int rowKV = 0; rowKV < qEnd; rowKV += bc) {

        //copy current (bc, hs) K/V block into smem
        for(int i = threadIdx.x; i < (bc*hs); i += blockDim.x) {
            int row = i / hs;
            int col = i % hs;
            k[i] = ((rowKV + row) < T) ? K[((rowKV + row) * stride_k_t) + (col * stride_k_hs)] : 0.0f;
            v[i] = ((rowKV + row) < T) ? V[((rowKV + row) * stride_v_t) + (col * stride_v_hs)] : 0.0f;
        }

        __syncthreads();

        for(int rowQ = warpIdx; rowQ < br; rowQ += num_warps) {
            //local register setup
            float local_max = -INFINITY;
            float row_m_prev = m[rowQ];
            float local_sum = 0.0f;

            //causal masking setup Q
            int absQRow = (outRow * br + rowQ);

            //calculate S=QK^T
            for(int colK = laneIdx; colK < bc; colK += 32) {
                //causal masking setup K
                int absKCol = rowKV + colK;
                //local register for sum
                local_sum = 0.0f;
                for(int i = 0; i < hs; i++) {
                    local_sum += q[rowQ * hs + i] * k[colK * hs + i]; 
                }
                local_sum *= sm_scale;
                float maskedVal = ((!causal || absQRow >= absKCol) && (absQRow < T && absKCol < T)) ? local_sum : -INFINITY;
                s[rowQ * bc + colK] = maskedVal;
                local_max = fmaxf(local_max, maskedVal);
            }

            
            //calculate warp level rowmax for the current (num_warps, bc) block
            float partial_max = row_max(local_max);
            float row_m_new = fmaxf(row_m_prev, partial_max);
            float alpha = expf(row_m_prev - row_m_new);

            //P=exp(S-m) and rowsum(P) setup
            local_sum = 0.0f;
            for(int col = laneIdx; col < bc; col += 32) {
                float temp = expf(s[rowQ * bc + col] - row_m_new);
                s[rowQ * bc + col] = temp;
                local_sum += temp;
            }

            //l_i(j) = (e^(m_i_j-1)-e^(m_i_j))*l_i(j-1)+rowsum(P)
            float partial_sum = row_sum(local_sum);
            if(laneIdx == 0) {
                l[rowQ] = alpha * l[rowQ] + partial_sum;
                m[rowQ] = row_m_new;
            }

            //finally calculate o_i(j)
            for(int colV = laneIdx; colV < hs; colV += 32) {
                local_sum = 0.0f;
                for(int i = 0; i < bc; i++) {
                    local_sum += s[rowQ * bc + i] * v[i * hs + colV];
                }
                //add the (num_warps, 32) dot product to rescaled o_i 
                o[rowQ * hs + colV] = (o[rowQ * hs + colV] * alpha) + local_sum;
            }
        }

        __syncthreads();
    }

    //epilogue: write final outputs to global memory
    for(int row = warpIdx; row < br; row += num_warps) {
        //absolute position of row in O
        int absORow = (outRow * br + row);
        if(absORow < T) {
            for(int col = laneIdx; col < hs; col += 32) {
                O[row * hs + col] = o[row * hs + col] / l[row];
            }
        }
        if(laneIdx == 0 && absORow < T) {
            L[row] = m[row] + logf(l[row]);
        }
    }
}

// ============= BACKWARD PASS =============

// ============= DELTA CALCULATION =============
__global__ void flash_attention_bwd_delta(const float* O, const float* dO, float* delta, int B, int nh, int T, int hs, 
    int stride_o_b, int stride_o_nh, int stride_o_t, int stride_o_hs,
    int stride_do_b, int stride_do_nh, int stride_do_t, int stride_do_hs) {
    int batch = blockIdx.z / nh;
    int head = blockIdx.z % nh;
    int outRow = blockIdx.y;
    int num_warps = blockDim.x / 32;
    int warpIdx = threadIdx.x / 32;
    int laneIdx = threadIdx.x % 32;

    __shared__ float smem[32];
    
    float local_sum = 0.0f;

    O += (batch * stride_o_b) + (head * stride_o_nh) + (outRow * stride_o_t);
    dO += (batch * stride_do_b) + (head * stride_do_nh) + (outRow * stride_do_t);
    delta += (batch * nh * T) + (head * T) + outRow;

    for(int col = threadIdx.x; col < hs; col += blockDim.x) {
        local_sum += O[col * stride_o_hs] * dO[col * stride_do_hs];
    }

    float partial_sum = row_sum(local_sum);

    if(laneIdx == 0) {
        smem[warpIdx] = partial_sum;
    }

    __syncthreads();

    if(warpIdx == 0) {
        float sum_val = (laneIdx < num_warps) ? smem[laneIdx] : 0.0f;
        float w0_sum = row_sum(sum_val);
        
        if(laneIdx == 0) {
            *delta = w0_sum;
        }
    }
}

// ============= DQ CALCULATION =============
template <int br, int bc, int hs>
__global__ void flash_attention_bwd_dq(const float* Q, const float* K, const float* V, const float* delta, const float* L, const float* dO, float* dQ, int B, int nh, int T, float sm_scale, bool causal,
    int stride_q_b, int stride_q_nh, int stride_q_t, int stride_q_hs,
    int stride_k_b, int stride_k_nh, int stride_k_t, int stride_k_hs,
    int stride_v_b, int stride_v_nh, int stride_v_t, int stride_v_hs,
    int stride_l_b, int stride_l_nh, int stride_l_t,
    int stride_do_b, int stride_do_nh, int stride_do_t, int stride_do_hs) 
{
    int batch = blockIdx.z / nh;
    int head = blockIdx.z % nh;
    int outRow = blockIdx.y;
    int num_warps = blockDim.x / 32;
    int warpIdx = threadIdx.x / 32;
    int laneIdx = threadIdx.x % 32;
    sm_scale = 1/sm_scale;

    //smem setup 
    __shared__ float q[br * hs];
    __shared__ float k[bc * hs];
    __shared__ float v[bc * hs];
    __shared__ float s[br * bc];
    __shared__ float _dO[br * hs];
    __shared__ float dP[br * bc];
    __shared__ float dS[br * bc];
    __shared__ float _dQ[br * hs];
    __shared__ float del[br];
    __shared__ float l[br];

    //starting pointers
    Q += (batch * stride_q_b) + (head * stride_q_nh) + (outRow * br * stride_q_t);
    K += (batch * stride_k_b) + (head * stride_k_nh);
    V += (batch * stride_v_b) + (head * stride_v_nh);
    dO += (batch * stride_do_b) + (head * stride_do_nh) + (outRow * br * stride_do_t);
    L += (batch * stride_l_b) + (head * stride_l_nh) + (outRow * br * stride_l_t);
    delta += (batch * nh * T) + (head * T) + (outRow * br);
    dQ += (batch * nh * T * hs) + (head * T * hs) + (outRow * br * hs);

    //copy q/dO into smem and setup dQ
    for(int i = threadIdx.x; i < (br*hs); i += blockDim.x) {
        int row = i / hs;
        int col = i % hs;
        int absQRow = (outRow * br + row);
        q[i] = (absQRow < T) ? Q[(row * stride_q_t) + (col * stride_q_hs)] : 0.0f;
        _dO[i] = (absQRow < T) ? dO[(row * stride_do_t) + (col * stride_do_hs)] : 0.0f;
        _dQ[i] = 0.0f;
    }

    __syncthreads();

    //copy L into smem
    for(int i = threadIdx.x; i < br; i += blockDim.x) {
        int absRow = outRow * br + i;
        l[i] = (absRow < T) ? L[i * stride_l_t] : 0.0f;
        del[i] = (absRow < T) ? delta[i] : 0.0f;
    }

    __syncthreads();

    //stage setup for causal masking Q
    int qEnd = causal ? min((outRow + 1) * br, T) : T;

    for(int rowKV = 0; rowKV < qEnd; rowKV += bc) {
        for(int i = threadIdx.x; i < (bc*hs); i += blockDim.x) {
            int row = i / hs;
            int col = i % hs;
            k[i] = ((rowKV + row) < T) ? K[((rowKV + row) * stride_k_t) + (col * stride_k_hs)] : 0.0f;
            v[i] = ((rowKV + row) < T) ? V[((rowKV + row) * stride_v_t) + (col * stride_v_hs)] : 0.0f;
        }

        __syncthreads();

        for(int rowQ = warpIdx; rowQ < br; rowQ += num_warps) {
            //local register setup
            float local_sum = 0.0f;

            //causal masking setup Q
            int absQRow = (outRow * br + rowQ);

            //calculate S=QK^T
            for(int colK = laneIdx; colK < bc; colK += 32) {
                //causal masking setup K
                int absKCol = rowKV + colK;
                //local register for sum
                local_sum = 0.0f;
                for(int i = 0; i < hs; i++) {
                    local_sum += q[rowQ * hs + i] * k[colK * hs + i]; 
                }
                local_sum *= sm_scale;
                float maskedVal = ((!causal || absQRow >= absKCol) && (absQRow < T && absKCol < T)) ? local_sum : -INFINITY;
                s[rowQ * bc + colK] = maskedVal;
            }

            //P=exp(S-L)
            for(int col = laneIdx; col < bc; col += 32) {
                float temp = expf(s[rowQ * bc + col] - l[rowQ]);
                s[rowQ * bc + col] = temp;
            }

            //calculate dP
            for(int colV = laneIdx; colV < bc; colV += 32) {
                local_sum = 0.0f;
                for(int i = 0; i < hs; i++) {
                    local_sum += _dO[rowQ * hs + i] * v[colV * hs + i];
                }
                //add the (num_warps, 32) dot product to rescaled o_i 
                dP[rowQ * bc + colV] = local_sum;
            }

            //calculate dS
            for(int col = laneIdx; col < bc; col += 32) {
                float temp = s[rowQ * bc + col] * (dP[rowQ * bc + col] - del[rowQ]);
                dS[rowQ * bc + col] = temp;
            }

            //calculate dQ and write to output
            for(int colK = laneIdx; colK < hs; colK += 32) {
                local_sum = 0.0f;
                for(int i = 0; i < bc; i++) {
                    local_sum += dS[rowQ * bc + i] * k[i * hs + colK];
                }
                //add the (num_warps, 32) dot product to rescaled o_i 
                _dQ[rowQ * hs + colK] += local_sum * sm_scale;
            }
        }

        __syncthreads();
    }

    //epilogue: write final outputs to global memory
    for(int row = warpIdx; row < br; row += num_warps) {
        //absolute position of row in dQ
        int absdQRow = (outRow * br + row);
        if(absdQRow < T) {
            for(int col = laneIdx; col < hs; col += 32) {
                dQ[row * hs + col] = _dQ[row * hs + col] ;
            }
        }
    }
}

// ============= DK/DV CALCULATION =============
template <int br, int bc, int hs>
__global__ void flash_attention_bwd_dk_dv(const float* Q, const float* K, const float* V, const float* delta, const float* L, const float* dO, float* dK, float* dV, int B, int nh, int T, float sm_scale, bool causal,
    int stride_q_b, int stride_q_nh, int stride_q_t, int stride_q_hs,
    int stride_k_b, int stride_k_nh, int stride_k_t, int stride_k_hs,
    int stride_v_b, int stride_v_nh, int stride_v_t, int stride_v_hs,
    int stride_l_b, int stride_l_nh, int stride_l_t,
    int stride_do_b, int stride_do_nh, int stride_do_t, int stride_do_hs) 
{
    int batch = blockIdx.z / nh;
    int head = blockIdx.z % nh;
    int outRow = blockIdx.y;
    int num_warps = blockDim.x / 32;
    int warpIdx = threadIdx.x / 32;
    int laneIdx = threadIdx.x % 32;
    sm_scale = 1/sm_scale;

    //smem setup 
    __shared__ float q[br * hs];
    __shared__ float k[bc * hs];
    __shared__ float v[bc * hs];
    __shared__ float s[br * bc];
    __shared__ float _dO[br * hs];
    __shared__ float dP[br * bc];
    __shared__ float dS[br * bc];
    __shared__ float _dK[bc * hs];
    __shared__ float _dV[bc * hs];
    __shared__ float del[br];
    __shared__ float l[br];

    //starting pointers
    Q += (batch * stride_q_b) + (head * stride_q_nh);
    K += (batch * stride_k_b) + (head * stride_k_nh) + (outRow * br * stride_k_t);
    V += (batch * stride_v_b) + (head * stride_v_nh) + (outRow * br * stride_v_t);
    dO += (batch * stride_do_b) + (head * stride_do_nh);
    L += (batch * stride_l_b) + (head * stride_l_nh);
    delta += (batch * nh * T) + (head * T);
    dK += (batch * nh * T * hs) + (head * T * hs) + (outRow * br * hs);
    dV += (batch * nh * T * hs) + (head * T * hs) + (outRow * br * hs);

    //copy q/dO into smem and setup dQ
    for(int i = threadIdx.x; i < (bc*hs); i += blockDim.x) {
        int row = i / hs;
        int col = i % hs;
        int absKVRow = (outRow * bc + row);
        k[i] = (absKVRow < T) ? K[(row * stride_k_t) + (col * stride_k_hs)] : 0.0f;
        v[i] = (absKVRow < T) ? V[(row * stride_v_t) + (col * stride_v_hs)] : 0.0f;
        _dK[i] = 0.0f;
        _dV[i] = 0.0f;
    }

    __syncthreads();

    int kvEnd = causal ? min((outRow + 1) * bc, T) : T;

    for(int rowQ = 0; rowQ < T; rowQ += br) {
        //copy q/dO into smem 
        for(int i = threadIdx.x; i < (br*hs); i += blockDim.x) {
            int row = i / hs;
            int col = i % hs;
            q[i] = ((rowQ + row) < T) ? Q[((rowQ + row) * stride_q_t) + (col * stride_q_hs)] : 0.0f;
            _dO[i] = ((rowQ + row) < T) ? dO[((rowQ + row) * stride_do_t) + (col * stride_do_hs)] : 0.0f; 
        }

        __syncthreads();

        //copy L into smem
        for(int i = threadIdx.x; i < br; i += blockDim.x) {
            l[i] = ((rowQ + i) < T) ? L[(rowQ + i) * stride_l_t] : 0.0f;
            del[i] = ((rowQ + i) < T) ? delta[(rowQ + i)] : 0.0f;
        }

        __syncthreads();

        //rowKV should actually be called colKV but technically K/V are being transposed so we are directly reading rows
        for(int rowKV = warpIdx; rowKV < bc; rowKV += num_warps) {
            //local register setup
            float local_sum = 0.0f;

            //causal masking setup Q
            int absKVRow = (outRow * bc + rowKV);

            //calculate S=QK^T
            for(int row = laneIdx; row < br; row += 32) {
                int absQRow = rowQ + row;
                local_sum = 0.0f;
                for(int i = 0; i < hs; i++) {
                    local_sum += q[row * hs + i] * k[rowKV * hs + i];
                }
                local_sum *= sm_scale;
                float maskedVal = ((!causal || absQRow >= absKVRow) && (absQRow < T && absKVRow < T)) ? local_sum : -INFINITY;
                s[row * bc + rowKV] = maskedVal;
            }

            //P=exp(S-L)
            for(int row = laneIdx; row < br; row += 32) {
                float temp = expf(s[row * bc + rowKV] - l[row]);
                s[row * bc + rowKV] = temp;
            }

            //calculate dV 
            for(int colH = laneIdx; colH < hs; colH += 32) {
                local_sum = 0.0f;
                for(int row = 0; row < br; row++) {
                    local_sum += s[row * bc + rowKV] * _dO[row * hs + colH];
                }
                _dV[rowKV * hs + colH] += local_sum;
            }

            //calculate dP
            for(int row = laneIdx; row < br; row += 32) {
                local_sum = 0.0f;
                for(int i = 0; i < hs; i++) {
                    local_sum += _dO[row * hs + i] * v[rowKV * hs + i];
                }
                dP[row * bc + rowKV] = local_sum;
            }

            //calculate dS
            for(int row = laneIdx; row < br; row += 32) {
                float temp = s[row * bc + rowKV] * (dP[row * bc + rowKV] - del[row]);
                dS[row * bc + rowKV] = temp;
            }

            //calculate dK
            for(int colH = laneIdx; colH < hs; colH += 32) {
                local_sum = 0.0f;
                for(int row = 0; row < br; row++) {
                    local_sum += dS[row * bc + rowKV] * q[row * hs + colH];
                }
                _dK[rowKV * hs + colH] += local_sum * sm_scale;
            }
        }

        __syncthreads();
    }

    //epilogue: write final outputs to global memory
    for(int row = warpIdx; row < bc; row += num_warps) {
        //absolute position of row in dQ
        int absdKVRow = (outRow * bc + row);
        if(absdKVRow < T) {
            for(int col = laneIdx; col < hs; col += 32) {
                dK[row * hs + col] = _dK[row * hs + col] ;
                dV[row * hs + col] = _dV[row * hs + col] ;
            }
        }
    }
}