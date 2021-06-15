import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def attention(Q, K, V, mask=None):
    dk = torch.tensor(K.shape[-1]).type(torch.float)
    score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (torch.sqrt(dk) + 1e-8)  # [n, n_head, step, step]
    if mask is not None:
        # change the value at masked position to negative infinity,
        # so the attention score at these positions after softmax will close to 0.
        score = score.masked_fill_(mask, -np.inf)
    atten = F.softmax(score, dim=-1)
    context = torch.matmul(atten, V)  # [n, num_head, step, head_dim]
    context = context.permute(0, 2, 1, 3)  # [n, step, num_head, head_dim]
    context = context.reshape((context.shape[0], context.shape[1], -1))
    return context  # [n, step, model_dim]


class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(32, 32)
        self.wk = nn.Linear(32, 32)
        self.wv = nn.Linear(32, 32)

        self.o_dense = nn.Linear(32, 32)
        self.o_drop = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(32)
        self.attention = None

    def forward(self, Q, K, V, mask):
        # residual connect
        original_Q = Q

        # linear projection
        key = self.wk(K)  # [n, step, num_heads * head_dim]
        value = self.wv(V)  # [n, step, num_heads * head_dim]
        query = self.wq(Q)  # [n, step, num_heads * head_dim]

        # split by head
        query = self.split_heads(query)  # [n, n_head, q_step, h_dim]
        key = self.split_heads(key)
        value = self.split_heads(value)  # [n, h, step, h_dim]
        context = attention(query, key, value, mask)  # [n, q_step, h*dv]
        o = self.o_dense(context)  # [n, step, dim]
        o = self.o_drop(o)

        o = self.layer_norm(original_Q + o)
        return o

    def split_heads(self, x):
        x = torch.reshape(x, (x.shape[0], x.shape[1], 4, 8))
        return x.permute(0, 2, 1, 3)




if __name__ == '__main__':
    import mask

    x = torch.ones(100, 11, 32)
    print(MultiHead()(x, x, x, mask.mask_x(torch.ones(100, 11))).shape)