import torch
from cuda_transformer.ops import flash_attention_fwd, flash_attention_bwd

class CudaFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        o, l = flash_attention_fwd(q, k, v, causal)
        ctx.save_for_backward(q, k, v, o, l)
        ctx.causal = causal
        return o, l

    @staticmethod
    def backward(ctx, grad_output, grad_l=None):
        q, k, v, o, l = ctx.saved_tensors
        causal = ctx.causal
        do = grad_output
        dq, dk, dv = flash_attention_bwd(q, k, v, o, do, l, causal)
        return dq, dk, dv, None