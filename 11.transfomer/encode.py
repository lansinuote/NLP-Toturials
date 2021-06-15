import torch
import torch.nn as nn

import atten
import fcout


class EncoderLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.mh = atten.MultiHead()
        self.fc = fcout.FullyConnectedOutput()

    def forward(self, x, mask):
        # 计算自注意力,维度不变
        # [b, 11, 32] -> [b, 11, 32]
        score = self.mh(x, x, x, mask)

        # 全连接输出,维度不变
        # [b, 11, 32] -> [b, 11, 32]
        out = self.fc(score)

        return out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = EncoderLayer()
        self.layer_2 = EncoderLayer()
        self.layer_3 = EncoderLayer()

    def forward(self, x, mask):
        x = self.layer_1(x, mask)
        x = self.layer_2(x, mask)
        x = self.layer_3(x, mask)
        return x


if __name__ == '__main__':
    import mask

    x = torch.ones(100, 11, 32)
    mask_x = mask.mask_x(torch.ones(100, 11))
    print(Encoder()(x, mask_x).shape)
