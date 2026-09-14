import torch
from torch import Tensor

__all__ = [
    "cuda_fma", "db_calculation", 
    "cuda_softmax", "cuda_softmax_bwd",
    "gelu_activation", "gelu_activation_bwd",
    "causal_self_attention_fwd", "causal_self_attention_bwd",
    "flash_attention_fwd", "flash_attention_bwd"
    ]

########### LINEAR LAYER ########### 
def cuda_fma(x: Tensor, weights: Tensor, bias: Tensor) -> Tensor:
    """performs fused multiply add operation in CUDA""" 
    return torch.ops.cuda_transformer.cuda_fma.default(x, weights, bias)

@torch.library.register_fake("cuda_transformer::cuda_fma")
def _(x, weights, bias):
    torch._check(x.dtype == torch.float32)
    torch._check(weights.dtype == torch.float32)
    torch._check(bias.dtype == torch.float32)
    torch._check(x.device == weights.device == bias.device)
    w = weights if weights.dim() == 3 else weights.unsqueeze(0)
    return torch.empty(x.size(0), x.size(1), w.size(2), dtype=x.dtype, device=x.device)

def db_calculation(dy: Tensor) -> Tensor:
    """"performs db calculation for backward pass"""
    return torch.ops.cuda_transformer.db_calculation.default(dy)

@torch.library.register_fake("cuda_transformer::db_calculation")
def _(dy):
    torch._check(dy.dtype == torch.float32)
    return torch.empty(dy.size(-1), dtype=dy.dtype, device=dy.device)

########### SOFTMAX ###########
def cuda_softmax(x: Tensor) -> Tensor:
    """performs softmax foward pass in cuda"""
    return torch.ops.cuda_transformer.cuda_softmax.default(x)

@torch.library.register_fake("cuda_transformer::cuda_softmax")
def _(x):
    torch._check(x.dtype == torch.float32)
    return torch.empty_like(x)

def cuda_softmax_bwd(probs: Tensor, dp: Tensor) -> Tensor:
    """performs softmax backward pass in cuda"""
    return torch.ops.cuda_transformer.cuda_softmax_bwd.default(probs, dp)

@torch.library.register_fake("cuda_transformer::cuda_softmax_bwd")
def _(probs, dp):
    torch._check(probs.dtype == torch.float32)
    torch._check(dp.dtype == torch.float32)
    torch._check(dp.device == probs.device)
    return torch.empty_like(probs)

########### GELU ###########
def gelu_activation(x: Tensor) -> Tensor:
    """performs gelu foward pass in cuda"""
    return torch.ops.cuda_transformer.gelu_activation.default(x)

@torch.library.register_fake("cuda_transformer::gelu_activation")
def _(x):
    torch._check(x.dtype == torch.float32)
    return torch.empty_like(x)

def gelu_activation_bwd(x: Tensor, dy: Tensor) -> Tensor:
    """performs gelu backward pass in cuda"""
    return torch.ops.cuda_transformer.gelu_activation_bwd.default(x, dy)

@torch.library.register_fake("cuda_transformer::gelu_activation_bwd")
def _(x, dy):
    torch._check(x.dtype == torch.float32)
    torch._check(dy.type == torch.float32)
    torch._check(dy.device == x.device)
    return torch.empty_like(x)

########### CAUSAL SELF ATTENTION LAYER ###########
def causal_self_attention_fwd(qkv: Tensor, nh: int, causal: bool) -> tuple[Tensor, Tensor]:
    """performs causal self attention forward pass in cuda"""
    return torch.ops.cuda_transformer.causal_self_attention_fwd.default(qkv, nh, causal)

@torch.library.register_fake("cuda_transformer::causal_self_attention_fwd")
def _(qkv, nh, causal):
    torch._check(qkv.dtype == torch.float32)
    b, t, three_c = qkv.shape
    c = three_c//3
    out = torch.empty((b, t, c), dtype=qkv.dtype, device=qkv.device)
    probs = torch.empty((b, t, t), dtype=qkv.dtype, device=qkv.device)
    return out, probs

def causal_self_attention_bwd(Q: Tensor, K: Tensor, V: Tensor, probs: Tensor, dO: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """performs causal self attention backward pass in cuda"""
    return torch.ops.cuda_transformer.causal_self_attention_bwd.default(Q, K, V, probs, dO)

@torch.library.register_fake("cuda_transformer::causal_self_attention_bwd")
def _(Q, K, V, probs, dO):
    torch._check(Q.dtype == torch.float32)
    torch._check(K.dtype == torch.float32) 
    torch._check(V.dtype == torch.float32)
    torch._check(probs.dtype == torch.float32)
    torch._check(dO.dtype == torch.float32)
    torch._check(Q.device == K.device == V.device == probs.device == dO.device)
    dq = torch.empty_like(Q)
    dk = torch.empty_like(K)
    dv = torch.empty_like(V)
    return dq, dk, dv

########### FLASH ATTENTION 2 LAYER ###########
def flash_attention_fwd(Q: Tensor, K: Tensor, V: Tensor, causal: bool) -> tuple[Tensor, Tensor]:
    """performs flash attention 2 foward pass in cuda"""
    return torch.ops.cuda_transformer.flash_attention_fwd.default(Q, K, V, causal)

@torch.library.register_fake("cuda_transformer::flash_attention_fwd")
def _(Q, K, V, causal):
    torch._check(Q.dtype == torch.float32)
    torch._check(K.dtype == torch.float32) 
    torch._check(V.dtype == torch.float32)
    torch._check(Q.device == K.device == V.device)
    O = torch.empty((Q.size(0), Q.size(1), Q.size(2), Q.size(3)), dtype=Q.dtype, device=Q.device)
    L = torch.empty((Q.size(0), Q.size(1), Q.size(2)), dtype=Q.dtype, device=Q.device)
    return O, L

def flash_attention_bwd(Q: Tensor, K: Tensor, V: Tensor, O: Tensor, dO: Tensor, L: Tensor, causal: bool):
    """performs flash attention 2 backward pass in cuda"""
    return torch.ops.cuda_transformer.flash_attention_bwd.default(Q, K, V, O, dO, L, causal)

@torch.library.register_fake("cuda_transformer::flash_attention_bwd")
def _(Q, K, V, O, dO, L, causal):
    torch._check(Q.dtype == torch.float32)
    torch._check(K.dtype == torch.float32) 
    torch._check(V.dtype == torch.float32)
    torch._check(O.dtype == torch.float32)
    torch._check(dO.dtype == torch.float32)
    torch._check(L.dtype == torch.float32)
    torch._check(Q.device == K.device == V.device == O.device == dO.device == L.device)
    dq = torch.empty_like(Q)
    dk = torch.empty_like(K)
    dv = torch.empty_like(V)
    return dq, dk, dv        