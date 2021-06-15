import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def attention(Q, K, V, mask):
    # b句话,每句话11个词,每个词编码成32维向量,4个头,每个头分到8维向量
    # Q,K,V = [b, 4, 11, 8]

    # [b, 4, 11, 8] * [b, 4, 8, 11] -> [b, 4, 11, 11]
    # Q,K矩阵相乘,结果除以根号头数,这里完全是公式
    score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(8)

    # mask遮盖,mask是true的地方都被替换成-inf
    # mask = [b, 1, 11, 11]
    score = score.masked_fill_(mask, -np.inf)
    score = F.softmax(score, dim=-1)

    # 这一步也是公式
    # [b, 4, 11, 11] * [b, 4, 11, 8] -> [b, 4, 11, 8]
    score = torch.matmul(score, V)

    # 每个头计算的结果合一
    # [b, 4, 11, 8] -> [b, 11, 32]
    score = score.permute(0, 2, 1, 3).reshape(-1, 11, 32)

    return score

class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_Q = nn.Linear(32, 32)
        self.fc_K = nn.Linear(32, 32)
        self.fc_V = nn.Linear(32, 32)

        self.out_fc = nn.Linear(32, 32)

        # 规范化之后,均值是0,标准差是1,前提是没有经过线性运算的话
        # mean = out.mean(dim=(0, 2))
        # std = out.std(dim=(0, 2))
        # BN是取不同样本的同一个通道的特征做归一化
        # LN取的是同一个样本的不同通道做归一化
        # affine=True,elementwise_affine=True,指定规范化后,再计算一个线性映射
        self.norm = nn.BatchNorm1d(num_features=11, affine=True)
        self.norm = nn.LayerNorm(normalized_shape=32, elementwise_affine=True)

    def forward(self, Q, K, V, mask):
        # b句话,每句话11个词,每个词编码成32维向量
        # Q,K,V = [b, 11, 32]
        b = Q.shape[0]

        # 保留下原始的Q,后面要做短接用
        original_Q = Q

        # 线性运算,维度不变
        # [b, 11, 32] -> [b, 11, 32]
        K = self.fc_K(K)
        V = self.fc_V(V)
        Q = self.fc_Q(Q)

        # 拆分成多个头
        # b句话,每句话11个词,每个词编码成32维向量,4个头,每个头分到8维向量
        # [b, 11, 32] -> [b, 4, 11, 8]
        Q = Q.reshape(b, 11, 4, 8).permute(0, 2, 1, 3)
        K = K.reshape(b, 11, 4, 8).permute(0, 2, 1, 3)
        V = V.reshape(b, 11, 4, 8).permute(0, 2, 1, 3)

        # 计算注意力
        # [b, 4, 11, 8] -> [b, 11, 32]
        score = attention(Q, K, V, mask)

        # 计算输出,维度不变
        # [b, 11, 32] -> [b, 11, 32]
        score = F.dropout(self.out_fc(score), 0.1)

        # 短接,规范化
        score = self.norm(original_Q + score)
        return score


if __name__ == '__main__':
    import mask

    x = torch.ones(100, 11, 32)
    print(MultiHead()(x, x, x, mask.mask_x(torch.ones(100, 11))).shape)