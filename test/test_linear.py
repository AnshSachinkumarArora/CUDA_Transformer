import torch
import torch.nn as nn
import torch.nn.functional as F
from cuda_transformer.ops import cuda_fma, db_calculation
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