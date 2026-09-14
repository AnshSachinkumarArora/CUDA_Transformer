import torch
import torch.nn as nn
import torch.nn.functional as F
from cuda_transformer.ops import gelu_activation, gelu_activation_bwd
import pytest

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