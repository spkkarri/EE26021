import torch

batch_size    = 32
block_size    = 256
max_iters     = 150_000
eval_interval = 500
learning_rate = 3e-4
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd        = 512
n_head        = 8
n_layer       = 8
dropout       = 0.1