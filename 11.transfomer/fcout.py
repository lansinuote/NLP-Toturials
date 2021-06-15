import torch
import torch.nn as nn


# 全连接输出层
class FullyConnectedOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=32, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.Dropout(p=0.1)
        )

        self.norm = nn.LayerNorm(normalized_shape=32, elementwise_affine=True)

    def forward(self, x):
        # 线性全连接运算
        # [b, 11, 32] -> [b, 11, 32]
        out = self.fc(x)

        # 做短接,正规化
        out = self.norm(x + out)

        return out


if __name__ == '__main__':
    x = torch.ones(100, 11, 32)
    print(FullyConnectedOutput()(x).shape)
