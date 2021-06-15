import torch
import torch.nn as nn

import data
import encode
import mask


class GPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(num_embeddings=4300, embedding_dim=256)
        self.seg_embed = nn.Embedding(num_embeddings=3, embedding_dim=256)

        self.position_embed = nn.Parameter(torch.randn(59, 256) / 10)

        self.encoder = encode.Encoder()

        self.fc_x_tail = nn.Linear(in_features=256, out_features=4300)
        self.fc_y = nn.Linear(in_features=59 * 256, out_features=2)

    def forward(self, x_head, seg):
        # [b, 60]
        mask_x = mask.get_key_padding_mask(x_head)

        # 编码,添加位置信息
        x_head = self.embed(x_head) + self.seg_embed(seg) + self.position_embed

        # 编码层计算
        # [b, 60, 256] -> [b, 60, 256]
        x_head = self.encoder(x_head, mask_x)

        x_tail = self.fc_x_tail(x_head)

        y = self.fc_y(x_head.reshape(x_head.shape[0], -1))

        return x_tail, y


model = GPT()
loss_func = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(1000):
    for i, (x, y, seg) in enumerate(data.get_dataloader()):
        # x = [b, 60]
        # y = [b]
        # seg = [b, 60]

        # x丢弃最后一位取头部,丢弃第一位取尾部
        # 实际的任务就是以头部预测尾部
        # [b, 60] -> [b, 59]
        x_head = x[:, :-1]
        x_tail = x[:, 1:]

        optim.zero_grad()

        # 模型计算
        # [b, 59],[b, 59] -> [b, 59, 4300],[b, 2]
        pred_x_tail, pred_y = model(x_head, seg[:, :-1])

        # 因为有两个计算结果,可以计算两份loss,再加权求和
        loss_x_tail = loss_func(pred_x_tail.reshape(-1, 4300), x_tail.reshape(-1))
        loss_y = loss_func(pred_y, y.reshape(-1))
        loss = loss_x_tail + loss_y * 0.2

        loss.backward()
        optim.step()

        if i % 10 == 0:
            # [b, 59, 4300] -> [b, 59]
            pred_x_tail = pred_x_tail.argmax(dim=2)

            # 把pad的位置排除掉,他们对计算正确率没有什么价值
            pred_x_tail[pred_x_tail == data.zidian['<PAD>']] = -1
            x_tail[x_tail == data.zidian['<PAD>']] = -2

            # 只计算非pad位置的正确率
            correct = (x_tail == pred_x_tail).sum().item()

            # 正确率的分母,是排除pad之后的位置数量
            total = (x_tail != -2).sum().item()

            print(epoch, i, loss.item(), correct / total)
            print(x_tail[0])
            print(pred_x_tail[0])