import torch
import math
from cuda_autograd_fns import cuda_flash_attention

# Setup hyperparams and tensors
B, H, N, D = 4, 32, 4096, 64
dtype = torch.float32
device = 'cuda'
causal = True
sm_scale = 1.0 / math.sqrt(D)

q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)

def run_kernel():
    return cuda_flash_attention.CudaFlashAttention.apply(q, k, v, causal)

# Kernel warmup
for _ in range(5):
    out, L = run_kernel()
    dO = torch.rand_like(out)
    out.backward(dO)
torch.cuda.synchronize()

# Forward pass
torch.cuda.profiler.start()
out, L = run_kernel()
torch.cuda.synchronize()
torch.cuda.profiler.stop()