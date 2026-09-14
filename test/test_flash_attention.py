import torch
import torch.nn as nn
import torch.nn.functional as F
from cuda_transformer.ops import flash_attention_fwd, flash_attention_bwd
import pytest

#flash attention 2 test
class TestFlashAttention:
    #naive attention for comparing L values since sdpa doesn't return L
    def torchCausalAttn(self, q, k, v, causal):
        sm_scale = k.shape[-1]**(-0.5)
        qkt = q @ k.transpose(-2, -1)
        qkt_scaled = qkt * sm_scale
        if causal:
            T = q.size(-2)
            causal_mask = torch.tril(torch.ones(T, T, device=q.device))
            qkt_scaled = qkt_scaled.masked_fill(causal_mask[:T, :T] == 0, float('-inf'))

        L = torch.logsumexp(qkt_scaled, dim=-1)

        return L
    
    #forward pass
    def flashAttnFwd(self, B, nh, T, hs, causal):
        q = torch.randn((B, nh, T, hs), device='cuda', dtype=torch.float32, requires_grad=True)
        k = torch.randn((B, nh, T, hs), device='cuda', dtype=torch.float32, requires_grad=True)
        v = torch.randn((B, nh, T, hs), device='cuda', dtype=torch.float32, requires_grad=True)
        q_cuda, k_cuda, v_cuda = q.detach().clone(), k.detach().clone(), v.detach().clone()
        sm_scale = k.shape[-1]**(-0.5)

        #get outputs
        torch_out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)
        torch_L = self.torchCausalAttn(q, k, v, causal)
        custom_out, custom_L = flash_attention_fwd(q_cuda, k_cuda, v_cuda, causal)

        return torch_out, torch_L, custom_out, custom_L, q, k, v, q_cuda, k_cuda, v_cuda

    #backward pass
    def flashAttnBwd(self, q, k, v, q_cuda, k_cuda, v_cuda, out, out_cuda, L, causal):
        #setup random output grads
        dO = torch.rand_like(out)

        #get torch outputs
        out.backward(dO)
        dq, dk, dv = q.grad, k.grad, v.grad

        #get kernel outputs
        dq_cuda, dk_cuda, dv_cuda = flash_attention_bwd(q_cuda, k_cuda, v_cuda, out_cuda, dO.detach().clone(), L.detach().clone(), causal)

        return dq, dk, dv, dq_cuda, dk_cuda, dv_cuda

    @pytest.mark.parametrize("B, nh, T, hs, causal", [
            (1, 1, 64,  128,  False),
            (1, 1, 64,  128,  True),
            (1, 1, 65,  64,  True),   # ragged tail
            (2, 4, 128, 32,  False),
            (2, 4, 128, 64,  True),
            (2, 4, 257, 64,  True),   # ragged tail
            (1, 2, 512, 128, True),
        ])
    def testFlashAttn(self, B, nh, T, hs, causal):
        #test forward pass for both O and L
        torch_out, torch_L, custom_out, custom_L, q, k, v, q_cuda, k_cuda, v_cuda = self.flashAttnFwd(B, nh, T, hs, causal)
        assert torch.allclose(custom_out, torch_out, atol=1e-5)
        assert torch.allclose(torch_L, custom_L, atol=1e-5)

        #test backward pass
        dq, dk, dv, dq_cuda, dk_cuda, dv_cuda = self.flashAttnBwd(q, k, v, q_cuda, k_cuda, v_cuda, torch_out, custom_out, torch_L, causal)
        assert torch.allclose(dq_cuda, dq, atol=1e-4)
        assert torch.allclose(dk_cuda, dk, atol=1e-4)
        assert torch.allclose(dv_cuda, dv, atol=1e-4)