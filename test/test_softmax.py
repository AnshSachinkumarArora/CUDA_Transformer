import torch
import torch.nn as nn
import torch.nn.functional as F
from cuda_transformer.ops import cuda_softmax, cuda_softmax_bwd
import pytest

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