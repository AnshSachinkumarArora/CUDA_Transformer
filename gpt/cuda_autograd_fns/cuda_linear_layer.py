import torch
from cuda_transformer.ops import cuda_fma, db_calculation

class CudaLinearLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, bias):
        #get output
        out = cuda_fma(x, weights, bias)
        #save inputs to context
        ctx.save_for_backward(x, weights, bias, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        #get context 
        x, weights, _, _ = ctx.saved_tensors
        dy = grad_output
        #get grads
        w = weights.unsqueeze(0) if len(weights.shape) == 2 else weights
        w = w.transpose(1,2)
        x_t = x.transpose(1, 2)
        zero_b_dx = torch.zeros((x.shape[-1]), dtype=torch.float32, device='cuda')
        dx = cuda_fma(dy, w, zero_b_dx)
        zero_b_dw = torch.zeros((dy.shape[-1]), dtype=torch.float32, device='cuda')
        dw = cuda_fma(x_t, dy, zero_b_dw)
        dw = torch.sum(dw, dim=0)
        db = db_calculation(dy)

        return dx, dw, db