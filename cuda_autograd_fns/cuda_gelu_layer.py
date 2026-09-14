import torch
from cuda_transformer.ops import gelu_activation, gelu_activation_bwd

class CudaGeluLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = gelu_activation(x)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        dx = gelu_activation_bwd(x, grad_output.contiguous())
        return dx