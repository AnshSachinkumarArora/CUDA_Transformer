import torch
from cuda_transformer.ops import cuda_softmax, cuda_softmax_bwd

class CudaSoftmaxLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = cuda_softmax(x)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        dp = grad_output.contiguous()
        dx = cuda_softmax_bwd(out, dp)
        return dx