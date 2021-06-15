import torch
import torch.nn as nn

import atten
import fcout


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.mh1 = atten.MultiHead()
        self.mh2 = atten.MultiHead()

        self.fc = fcout.FullyConnectedOutput()

    def forward(self, x, y, mask_x, mask_y):
        # 先计算y的自注意力,维度不变
        # [b, 11, 32] -> [b, 11, 32]
        y = self.mh1(y, y, y, mask_y)

        # 结合x和y的注意力计算,维度不变
        # [b, 11, 32],[b, 11, 32] -> [b, 11, 32]
        y = self.mh2(y, x, x, mask_x)

        # 全连接输出,维度不变
        # [b, 11, 32] -> [b, 11, 32]
        y = self.fc(y)

        return y


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = DecoderLayer()
        self.layer_2 = DecoderLayer()
        self.layer_3 = DecoderLayer()

    def forward(self, x, y, mask_x, mask_y):
        y = self.layer_1(x, y, mask_x, mask_y)
        y = self.layer_2(x, y, mask_x, mask_y)
        y = self.layer_3(x, y, mask_x, mask_y)
        return y


if __name__ == '__main__':
    import mask

    x = torch.ones(100, 11, 32)
    y = torch.ones(100, 11, 32)
    mask_x = mask.mask_x(torch.ones(100, 11))
    mask_y = mask.mask_y(torch.ones(100, 11))
    print(Decoder()(x, y, mask_x, mask_y).shape)
