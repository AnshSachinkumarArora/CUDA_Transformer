import torch
import torch.nn as nn
import torch.nn.functional as F
from cuda_transformer.ops import cuda_fma, db_calculation, cuda_softmax, cuda_softmax_bwd, gelu_activation, gelu_activation_bwd, causal_self_attention_fwd, causal_self_attention_bwd, flash_attention_fwd, flash_attention_bwd
import pytest

#linear layer test
class TestLinearLayer:
    #forward pass
    def linearLayerFwd(self, B, T, in_features, out_features):
        #setup inputs
        x = torch.rand((B, T, in_features), dtype=torch.float32, device='cuda', requires_grad=True)
        x_cuda = x.clone().detach()
        
        #setup torch linear layer
        torch_linear = nn.Linear(in_features, out_features, dtype=torch.float32, device='cuda')

        #get weights/bias to be reused within cuda kernel
        weights_cuda = torch_linear.weight.clone().detach().t()
        bias_cuda = torch_linear.bias.clone().detach()
        weights = torch_linear.weight
        bias = torch_linear.bias
        print(f'weight shape {weights_cuda.shape}, bias shape {bias_cuda.shape}')

        custom_out = cuda_fma(x_cuda, weights_cuda, bias_cuda)
        torch_out = torch_linear(x)

        return custom_out, torch_out, x, weights, bias, x_cuda, weights_cuda, bias_cuda

    #backward pass
    def linearLayerBwd(self, x, weights, bias, out, x_cuda, weights_cuda):
        #generate random dy
        dy = torch.rand_like(out)

        #get autograd grads
        dx, dw, db = torch.autograd.grad(
            outputs=out, inputs=[x, weights, bias], grad_outputs=dy
        )

        #get cuda kernel grads
        w = weights_cuda.unsqueeze(0) if len(weights_cuda.shape) == 2 else weights_cuda
        assert len(w.shape) == 3
        w = w.transpose(1,2)
        zero_b = torch.zeros((x.shape[2]), dtype=torch.float32, device='cuda')
        dy_t = dy.transpose(1, 2)
        dx_cuda = cuda_fma(dy, w, zero_b)
        dw_cuda = cuda_fma(dy_t, x_cuda, zero_b)
        dw_cuda = torch.sum(dw_cuda, dim=0)
        db_cuda = db_calculation(dy)

        return dx, dw, db, dx_cuda, dw_cuda, db_cuda

    @pytest.mark.parametrize("B, T, in_features, out_features", [
        (1, 32, 64,  32),
        (1, 32, 65,  32),
        (2, 64, 32, 64),
        (2, 128, 32, 64),
        (1, 512, 70, 120),
    ])
    def testLinearLayer(self, B, T, in_features, out_features):
        #test forward pass
        custom_out, torch_out, x, weights, bias, x_cuda, weights_cuda, bias_cuda = self.linearLayerFwd(B, T, in_features, out_features)
        assert torch.allclose(custom_out, torch_out, atol=1e-5)

        #test backward pass
        dx, dw, db, dx_cuda, dw_cuda, db_cuda = self.linearLayerBwd(x, weights, bias, torch_out, x_cuda, weights_cuda)
        assert torch.allclose(dx, dx_cuda, atol=1e-5)
        assert torch.allclose(dw, dw_cuda, atol=1e-5)
        assert torch.allclose(db, db_cuda, atol=1e-5)

#softmax kernel test
class TestSoftmax:
    #forward pass
    def softmaxFwd(self, batch, m, n):
        #setup inputs
        x = torch.rand((batch, m, n), dtype=torch.float32, device='cuda', requires_grad=True)
        x_cuda = x.clone().detach()

        #get outputs
        torch_out = F.softmax(x, dim=-1)
        custom_out = cuda_softmax(x_cuda)

        return torch_out, custom_out, x

    #backward pass 
    def softmaxBwd(self, probs, x):
        #setup incoming grads
        dp = torch.rand_like(probs)

        #get grads
        probs.backward(dp)
        ds = x.grad
        ds_cuda = cuda_softmax_bwd(probs, dp)

        return ds, ds_cuda

    @pytest.mark.parametrize("batch, m, n", [
            (1, 64, 32),
            (1, 65, 32),
            (2, 32, 64),
            (4, 128, 64),
            (1, 70, 120),
        ])
    def testSoftmax(self, batch, m, n):
        #test forward pass
        torch_out, custom_out, x = self.softmaxFwd(batch, m, n)
        assert torch.allclose(custom_out, torch_out, atol=1e-5)

        #test backward pass
        ds, ds_cuda = self.softmaxBwd(torch_out, x)
        assert torch.allclose(ds, ds_cuda, atol=1e-5)

#gelu kernel test
class TestGeLU:
    #forward pass
    def geluFwd(self, batch, m, n):
        #setup inputs
        x = torch.rand((batch, m, n), dtype=torch.float32, device='cuda', requires_grad=True)
        x_cuda = x.clone().detach()

        #get outputs
        torch_out = F.gelu(x)
        custom_out = gelu_activation(x_cuda)

        #inspect some of the outputs
        # print(f'\n===== TORCH OUTPUTS =====\n{torch_out[:, :2, :5]} .... \n')
        # print(f'===== KERNEL OUTPUTS =====\n{custom_out[:, :2, :5]} .... \n')

        return torch_out, custom_out, x

    def geluBwd(self, x, out):
        #setup incoming grads
        dy = torch.rand_like(x)

        #get grads
        out.backward(dy)
        dx = x.grad
        dx_cuda = gelu_activation_bwd(x, dy)

        return dx, dx_cuda

    @pytest.mark.parametrize("batch, m, n", [
            (1, 64, 32),
            (1, 65, 32),
            (2, 32, 64),
            (4, 128, 64),
            (1, 70, 120),
        ])
    def testGelu(self, batch, m, n):
        #test forward pass
        torch_out, custom_out, x = self.geluFwd(batch, m, n)
        assert torch.allclose(custom_out, torch_out, atol=1e-5)

        #test backward pass
        dx, dx_cuda = self.geluBwd(x, torch_out)
        assert torch.allclose(dx_cuda, dx, atol=1e-5)

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
