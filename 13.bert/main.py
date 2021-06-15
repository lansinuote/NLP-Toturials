import torch
import torch.nn as nn

import data
import encode
import mask
import replace


class BERT(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(num_embeddings=4301, embedding_dim=256)
        self.seg_embed = nn.Embedding(num_embeddings=3, embedding_dim=256)

        self.position_embed = nn.Parameter(torch.randn(60, 256) / 10)

        self.encoder = encode.Encoder()

        self.fc_x_tail = nn.Linear(in_features=256, out_features=4301)
        self.fc_y = nn.Linear(in_features=60 * 256, out_features=2)

    def forward(self, replace_x, seg):
        # [b, 60]
        mask_x = mask.get_key_padding_mask(replace_x)

        # 编码,添加位置信息
        # [b, 60] -> [b, 60, 256]
        replace_x = self.embed(replace_x) + self.seg_embed(seg) + self.position_embed

        # 编码层计算
        # [b, 60, 256] -> [b, 60, 256]
        replace_x = self.encoder(replace_x, mask_x)

        # 全连接层计算
        # [b, 60, 256] -> [b, 60, 4301]
        x_tail = self.fc_x_tail(replace_x)
        # [b, 60*256] -> [b, 2]
        y = self.fc_y(replace_x.reshape(replace_x.shape[0], -1))

        return x_tail, y


model = BERT()
loss_func = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(1000):
    for i, (x, y, seg) in enumerate(data.get_dataloader()):
        # x = [b, 60]
        # y = [b]
        # seg = [b, 60]

        # 随机替换x中的某些字符,replace是否被操作过的矩阵,这里的操作包括不替换
        # replace_x = [b, 60]
        # replace = [b, 60]
        replace_x, replace_mask = replace.random_replace(x)

        optim.zero_grad()

        # 模型计算
        # [b, 60],[b, 60] -> [b, 60, 4301],[b, 2]
        pred_x, pred_y = model(replace_x, seg)

        # 只把被操作过的字提取出来
        # [b, 60, 4301] -> [操作过的字数量 * 4301]
        pred_x = torch.masked_select(pred_x, replace_mask.unsqueeze(2))
        # 整理成n个字
        # [操作过的字数量 * 4301] -> [操作过的字数量, 4301]
        pred_x = pred_x.reshape(-1, 4301)

        # 把被操作之前的字取出来
        # [b, 60] -> [操作过的字数量]
        x = torch.masked_select(x, replace_mask)

        # 因为有两个计算结果,可以计算两份loss,再加权求和
        loss_x = loss_func(pred_x, x)
        loss_y = loss_func(pred_y, y.reshape(-1))
        loss = loss_x + loss_y * 0.2

        loss.backward()
        optim.step()

        if i % 10 == 0:
            # [操作过的字数量 * 4301] -> [操作过的字数量]
            pred_x = pred_x.argmax(dim=1)

            # 只计算操作过的位置的正确率
            correct = (x == pred_x).sum().item()

            total = len(x)
            print(epoch, i, loss.item(), correct / total)
            print(x[:50])
            print(pred_x[:50])
