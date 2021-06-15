import numpy as np
import torch
import torch.nn as nn


class PositionEmbedding(nn.Module):
    def __init__(self, max_len=11, emb_dim=32, n_vocab=27):
        super().__init__()
        pos = np.expand_dims(np.arange(max_len), 1)  # [max_len, 1]
        pe = pos / np.power(1000, 2 * np.expand_dims(np.arange(emb_dim) // 2, 0) / emb_dim)  # [max_len, emb_dim]
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = np.expand_dims(pe, 0)  # [1, max_len, emb_dim]
        self.pe = torch.from_numpy(pe).type(torch.float32)
        self.embeddings = nn.Embedding(n_vocab, emb_dim)
        self.embeddings.weight.data.normal_(0, 0.1)

    def forward(self, x):
        device = self.embeddings.weight.device
        self.pe = self.pe.to(device)
        x_embed = self.embeddings(x.long())
        x_embed = x_embed + self.pe  # [n, step, emb_dim]
        return x_embed  # [n, step, emb_dim]


if __name__ == '__main__':
    import data

    print(PositionEmbedding()(data.get_sample()[0]).shape)