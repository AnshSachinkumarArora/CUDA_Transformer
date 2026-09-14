import torch
from cuda_transformer.ops import causal_self_attention_fwd, causal_self_attention_bwd

class CudaCausalSelfAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, nh, causal):
        out, probs = causal_self_attention_fwd(qkv, nh, causal)
        ctx.save_for_backward(qkv, probs)
        ctx.nh = nh
        return out

    @staticmethod
    def backward(ctx, grad_output):
        qkv, probs = ctx.saved_tensors
        nh = ctx.nh

        q, k, v = qkv.chunk(3, dim=-1)
        B, T, C = q.shape
        hs = C//nh

        dO = grad_output.contiguous()
        sm_scale = hs**(0.5)
        q = q.reshape(B, T, nh, hs).transpose(1, 2).reshape(B * nh, T, hs)
        k = k.reshape(B, T, nh, hs).transpose(1, 2).reshape(B * nh, T, hs)
        v = v.reshape(B, T, nh, hs).transpose(1, 2).reshape(B * nh, T, hs)
        dO = dO.reshape(B, T, nh, hs).transpose(1, 2).reshape(B * nh, T, hs)
        probs = probs.reshape(B * nh, T, T)

        dq, dk, dv = causal_self_attention_bwd(q, k, v, probs, dO)

        dq = dq.reshape(B, nh, T, hs).transpose(1, 2).reshape(B, T, C)
        dk = dk.reshape(B, nh, T, hs).transpose(1, 2).reshape(B, T, C)
        dv = dv.reshape(B, nh, T, hs).transpose(1, 2).reshape(B, T, C)
        grads = torch.concat([dq, (dk/sm_scale), (dv/sm_scale)], dim=-1)

        return grads, None, None