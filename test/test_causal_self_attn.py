import torch
import torch.nn as nn
import torch.nn.functional as F
from cuda_transformer.ops import causal_self_attention_fwd, causal_self_attention_bwd
import pytest

#causal self attention test
class TestCausalSelfAttention:
    #pytorch reference causal self attention implementation
    def torchCausalAttn(self, qkv, B, T, nh, hs, causal):
        C = nh * hs
        q = qkv[:, :, :C]
        k = qkv[:, :, C:2*C]
        v = qkv[:, :, 2*C:]

        q = q.reshape(B, T, nh, hs).transpose(1, 2)
        k = k.reshape(B, T, nh, hs).transpose(1, 2)
        v = v.reshape(B, T, nh, hs).transpose(1, 2)
        sm_scale = k.shape[-1]**(-0.5)

        causal_mask = torch.tril(torch.ones(T, T, device=qkv.device))
        qkt = q @ k.transpose(-2, -1)
        qkt_scaled = qkt * sm_scale
        if causal:
            qkt_scaled = qkt_scaled.masked_fill(causal_mask[:T, :T] == 0, float('-inf'))
        qkt_softmax = F.softmax(qkt_scaled, dim=-1)
        scaled_dot_attn = qkt_softmax @ v
        scaled_dot_attn = scaled_dot_attn.transpose(1, 2).reshape(B, T, C)

        return scaled_dot_attn, qkt_softmax

    #forward pass
    def causalAttnFwd(self, B, T, nh, hs, causal):
        #setup inputs
        qkv = torch.rand((B, T, 3*nh*hs), dtype=torch.float32, device='cuda', requires_grad=True)
        qkv_cuda = qkv.clone().detach()

        #get outputs
        torch_out, probs = self.torchCausalAttn(qkv, B, T, nh, hs, causal)
        custom_out, custom_probs = causal_self_attention_fwd(qkv_cuda, nh, causal)

        return torch_out, custom_out, probs, custom_probs, qkv, qkv_cuda

    #backward pass
    def causalAttnBwd(self, qkv, qkv_cuda, out, probs, B, T, nh, hs):
        #generate random output grads
        dO = torch.rand_like(out)
        sm_scale = hs**(0.5)

        #get torch grads
        out.backward(dO)
        dq, dk, dv = qkv.grad.chunk(3, dim=-1)
        dq = dq.reshape(B, T, nh, hs).transpose(1, 2).reshape(B * nh, T, hs)
        dk = dk.reshape(B, T, nh, hs).transpose(1, 2).reshape(B * nh, T, hs)
        dv = dv.reshape(B, T, nh, hs).transpose(1, 2).reshape(B * nh, T, hs)

        #get custom grads
        q_cuda, k_cuda, v_cuda = qkv_cuda.chunk(3, dim=-1)
        q_cuda = q_cuda.reshape(B, T, nh, hs).transpose(1, 2).reshape(B * nh, T, hs)
        k_cuda = k_cuda.reshape(B, T, nh, hs).transpose(1, 2).reshape(B * nh, T, hs)
        v_cuda = v_cuda.reshape(B, T, nh, hs).transpose(1, 2).reshape(B * nh, T, hs)
        #reshape dO and probs for kernel 
        dO = dO.reshape(B, T, nh, hs).transpose(1, 2).reshape(B * nh, T, hs)
        probs = probs.reshape(B * nh, T, T)
        dq_cuda, dk_cuda, dv_cuda = causal_self_attention_bwd(q_cuda, k_cuda, v_cuda, probs, dO)

        return dq, dk, dv, (dq_cuda/sm_scale), (dk_cuda/sm_scale), dv_cuda

    @pytest.mark.parametrize("B, T, nh, hs, causal", [
            (1, 64,  1, 32,  False),
            (1, 64,  1, 32,  True),
            (1, 65,  1, 32,  True),   # ragged tail
            (2, 128, 4, 64,  False),
            (2, 128, 4, 64,  True),
            (2, 257, 4, 64,  True),   # ragged tail
            (1, 512, 2, 128, True),
        ])
    def testCausalSelfAttn(self, B, T, nh, hs, causal):
        #test forward pass
        torch_out, custom_out, probs, custom_probs, qkv, qkv_cuda = self.causalAttnFwd(B, T, nh, hs, causal)
        assert torch.allclose(custom_out, torch_out, atol=1e-5)

        #test backward pass
        dq, dk, dv, dq_cuda, dk_cuda, dv_cuda = self.causalAttnBwd(qkv, qkv_cuda, torch_out, custom_probs, B, T, nh, hs)

        # print(f'\ndq shape {dq.shape}')
        # print(f'dk shape {dk.shape}')
        # print(f'dv shape {dv.shape}')
        # print(f'dq_cuda shape {dq_cuda.shape}')
        # print(f'dk_cuda shape {dk_cuda.shape}')
        # print(f'dv_cuda shape {dv_cuda.shape}\n')

        #atol 1e-4 here because FP32 accumulation drift is higher in backward passes
        assert torch.allclose(dq_cuda, dq, atol=1e-4)
        assert torch.allclose(dk_cuda, dk, atol=1e-4)
        assert torch.allclose(dv_cuda, dv, atol=1e-4)