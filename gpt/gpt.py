import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.adamw import AdamW
torch.manual_seed(117)
import numpy as np
import tiktoken
from cuda_autograd_fns import cuda_causal_self_attention, cuda_flash_attention, cuda_gelu_layer, cuda_linear_layer, cuda_softmax_layer
import time

if not torch.cuda.is_available():
    raise RuntimeError("Aborting run: CUDA is required")
    
device = 'cuda'

#hyperparams
vocab_size = 100256
d_model = 512 #C
num_heads = 4 #nh
block_size = 256 #T
batch_size = 64 #B
head_dim = d_model//num_heads #hs
mini_batch_size = 4
num_iters = 1000
num_blocks = 1
learning_rate = 6e-4
dataset_path = 'dataset/dataset.bin'
encoder = tiktoken.get_encoding('cl100k_base')
max_seq_len = 1024

#dataset creation
dataset = np.memmap(dataset_path, dtype=np.uint32, mode='r')
#train/val split
split_size = int(0.9*len(dataset))
train = dataset[:split_size]
val = dataset[split_size:]

#data loader
def Generate_Batch(split):
    data = train if split == 'train' else val
    ix = torch.randint(len(data) - block_size, (mini_batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y

class LayerNorm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.epsilon = 1e-5

    def forward(self, idx):
        idx_var = torch.var(idx, dim=-1, keepdim=True, unbiased=False)
        idx_mean = torch.mean(idx, dim=-1, keepdim=True)
        idx_normalized = (idx - idx_mean)/torch.sqrt(idx_var + self.epsilon)
        output = idx_normalized * self.alpha + self.beta
        return output

class Embedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = nn.Embedding(block_size, d_model)

    def forward(self, tokens):
        T = tokens.shape[1]
        embedded = self.embedding(tokens)
        positions = self.positional(torch.arange(T, device=device))
        combined = embedded + positions
        return combined

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model//num_heads
        
        #QKV projection weights and biases
        self.qkv_weight = nn.Parameter(torch.empty(d_model, 3 * d_model))
        self.qkv_bias = nn.Parameter(torch.zeros(3 * d_model))

        #Output projection weights and biases
        self.out_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.out_bias = nn.Parameter(torch.zeros(d_model))
        
        #Apply standard Transformer weight initialization
        nn.init.normal_(self.qkv_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_weight, mean=0.0, std=0.02)

    def forward(self, tokens, use_flash_attention=False):
        #get dims
        B, T, C = tokens.shape
        
        #get QKV
        qkv = cuda_linear_layer.CudaLinearLayer.apply(tokens, self.qkv_weight, self.qkv_bias) #(B, T, 3C)

        #calculate attention
        if not use_flash_attention:
            attention_out = cuda_causal_self_attention.CudaCausalSelfAttention.apply(qkv, self.num_heads, True) #(B, T, C)
        else:
            q, k, v = qkv.chunk(3, dim=-1) #(B, T, C) each
            q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            attention_out, _ = cuda_flash_attention.CudaFlashAttention.apply(q, k, v, True) #(B, nh, T, hs)
            attention_out = attention_out.transpose(1, 2).reshape(B, T, C)

        #calculate output
        output = cuda_linear_layer.CudaLinearLayer.apply(attention_out, self.out_weight, self.out_bias) #(B, T, C)

        return output

class MLP(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.d_model = d_model
        #setup fc1
        self.fc1_weights = nn.Parameter(torch.empty(d_model, 4 * d_model))
        self.fc1_bias = nn.Parameter(torch.zeros(4 * d_model))

        #setup fc2
        self.fc2_weights = nn.Parameter(torch.empty(4 * d_model, d_model))
        self.fc2_bias = nn.Parameter(torch.zeros(d_model))

        #init weights
        nn.init.normal_(self.fc1_weights, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2_weights, mean=0.0, std=0.02)
        

    def forward(self, tokens):
        tokens = cuda_linear_layer.CudaLinearLayer.apply(tokens, self.fc1_weights, self.fc1_bias)
        tokens = cuda_gelu_layer.CudaGeluLayer.apply(tokens)
        tokens = cuda_linear_layer.CudaLinearLayer.apply(tokens, self.fc2_weights, self.fc2_bias)
        return tokens
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads) -> None:
        super().__init__()
        self.ln1 = LayerNorm()
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ln2 = LayerNorm()
        self.mlp = MLP(d_model)

    def forward(self, x, use_flash_attention):
        x = x + self.attn(self.ln1(x), use_flash_attention)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, num_blocks, d_model, num_heads, vocab_size) -> None:
        super().__init__()
        self.embedding = Embedding()
        self.mha = nn.ModuleList([DecoderBlock(d_model, num_heads) for _ in range(num_blocks)])
        self.ln = LayerNorm()
        self.lm_head_weights = nn.Parameter(torch.empty(d_model, vocab_size))
        self.lm_head_bias = nn.Parameter(torch.zeros(vocab_size))
        nn.init.normal_(self.lm_head_weights, mean=0.0, std=0.02)

    def forward(self, x, y=None, use_flash_attention=False):
        x = self.embedding(x)
        for layer in self.mha:
            x = layer(x, use_flash_attention)
        x = self.ln(x)
        logits = cuda_linear_layer.CudaLinearLayer.apply(x, self.lm_head_weights, self.lm_head_bias)

        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)
            y = y.reshape(B*T)
            loss = F.cross_entropy(logits, y)
        
        return logits, loss
    
    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx = idx if idx.shape[-1] <= block_size else idx[:, -block_size:]
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = cuda_softmax_layer.CudaSoftmaxLayer.apply(logits.unsqueeze(1)).squeeze(1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPT(num_blocks, d_model, num_heads, vocab_size)
model = model.to(device)

#training loop
optimizer = AdamW(model.parameters(), lr=learning_rate)
#splitting into mini_batches due to gpu memory constraints
grad_steps = int(batch_size/mini_batch_size)

#time training loop
start_time = time.perf_counter()

for step in range(num_iters):
    optimizer.zero_grad(set_to_none=True)
    loss_accumulator = 0.0
    for _ in range(grad_steps):
        xb, yb = Generate_Batch('train')
        logits, loss = model(xb, yb, use_flash_attention=True)
        loss = loss/grad_steps
        loss.backward()
        loss_accumulator += loss
    if step % 100 == 0: print(f'the loss is {loss_accumulator} on step {step}')
    optimizer.step()

end_time = time.perf_counter()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")
print(encoder.decode(model.generate(idx=torch.zeros((1,1), dtype=torch.long, device=device), max_tokens=min(max_seq_len, 256))[0].tolist()))