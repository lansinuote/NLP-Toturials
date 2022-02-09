import torch
import torch.nn as nn

import data
import decode
import encode
import mask
import position


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_x = position.PositionEmbedding()
        self.embed_y = position.PositionEmbedding()
        self.encoder = encode.Encoder()
        self.decoder = decode.Decoder()
        self.fc_out = nn.Linear(32, 27)

    def forward(self, x, y):
        # [b, 1, 11, 11]
        mask_x = mask.mask_x(x)
        mask_y = mask.mask_y(y)

        # 编码,添加位置信息
        # x = [b, 11] -> [b, 11, 32]
        # y = [b, 11] -> [b, 11, 32]
        x, y = self.embed_x(x), self.embed_y(y)

        # 编码层计算
        # [b, 11, 32] -> [b, 11, 32]
        x = self.encoder(x, mask_x)

        # 解码层计算
        # [b, 11, 32],[b, 11, 32] -> [b, 11, 32]
        y = self.decoder(x, y, mask_x, mask_y)

        # 全连接输出,维度不变
        # [b, 11, 32] -> [b, 11, 32]
        y = self.fc_out(y)

        return y


def predict(x):
    # x = [b, 11]
    model.eval()

    # [b, 1, 11, 11]
    mask_x = mask.mask_x(x)

    # 初始化输出,这个是固定值
    # [b, 12]
    # [[25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    target = [data.zidian["<SOS>"]] + [data.zidian["<PAD>"]] * 11
    target = torch.LongTensor(target).unsqueeze(0)

    # x编码,添加位置信息
    # [b, 11] -> [b, 11, 32]
    x = model.embed_x(x)

    # 编码层计算,维度不变
    # [b, 11, 32] -> [b, 11, 32]
    x = model.encoder(x, mask_x)

    # 遍历生成第0个词到第11个词
    for i in range(11):
        # 丢弃target中的最后一个词
        # 因为计算时,是以当前词,预测下一个词,所以最后一个词没有用
        # [b, 11]
        y = target[:, :-1]

        # [b, 1, 11, 11]
        mask_y = mask.mask_y(y)

        # y编码,添加位置信息
        # [b, 11] -> [b, 11, 32]
        y = model.embed_y(y)

        # 解码层计算,维度不变
        # [b, 11, 32],[b, 11, 32] -> [b, 11, 32]
        y = model.decoder(x, y, mask_x, mask_y)

        # 全连接输出,27分类
        # [b, 11, 32] -> [b, 11, 27]
        out = model.fc_out(y)

        # 取出当前词的输出
        # [b, 11, 27] -> [b, 27]
        out = out[:, i, :]

        # 取出分类结果
        # [b, 27] -> [b]
        out = out.argmax(dim=1).detach()

        # 以当前词预测下一个词,填到结果中
        target[:, i + 1] = out

    model.train()
    return target


model = Transformer()
loss_func = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.002)
for i in range(100):
    for batch_i, (x, y) in enumerate(data.get_dataloader()):
        # x = [b, 11]
        # x = 05-06-15<PAD><PAD><PAD>
        # y = [b, 12]
        # y = <SOS>15/Jun/2005<EOS><PAD>

        optim.zero_grad()

        # 在训练时,是拿y的每一个字符输入,预测下一个字符,所以不需要最后一个字典
        pred = model(x, y[:, :-1])

        loss = loss_func(pred.reshape(-1, 27), y[:, 1:].reshape(-1))
        loss.backward()
        optim.step()

        if batch_i % 50 == 0:
            pred = data.seq_to_str(predict(x[0:1])[0])
            print(i, data.seq_to_str(x[0]), data.seq_to_str(y[0]), pred)
