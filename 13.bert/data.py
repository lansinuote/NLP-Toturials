import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

zidian = {}
with open('../data/msr_paraphrase/zidian.txt') as fr:
    for line in fr.readlines():
        k, v = line.split(' ')
        zidian[k] = int(v)

zidian['<MASK>'] = len(zidian)


# 定义数据
class MsrDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv('../data/msr_paraphrase/数字化数据.txt', nrows=2000)

    def __getitem__(self, i):
        return self.data.iloc[i]

    def __len__(self):
        return len(self.data)


def to_tensor(data):
    b = len(data)
    # N句话,每句话30个词
    xs = np.zeros((b, 60))
    ys = np.zeros(b)
    segs = np.zeros((b, 60))

    for i in range(b):
        same, s1, s2 = data[i]

        ys[i] = same

        # 添加首尾符号
        s1 = [zidian['<SOS>']] + s1.split(',')[:28] + [zidian['<EOS>']]
        s2 = s2.split(',')[:28] + [zidian['<EOS>']]
        x = s1 + s2

        # 在前面补0到统一长度
        x = x[::-1] + [zidian['<PAD>']] * 60
        x = x[:60]
        x = x[::-1]

        xs[i] = x

        # 形状和x一样,但是内容不一样
        # 补0的地方是0,s1的地方是1,s2的地方是2
        seg = [0] * (60 - len(s1) - len(s2))
        seg += [1] * len(s1)
        seg += [2] * len(s2)
        segs[i] = seg

    return torch.LongTensor(xs), torch.LongTensor(ys), torch.LongTensor(segs)


# 数据加载器
def get_dataloader():
    dataloader = DataLoader(dataset=MsrDataset(),
                            batch_size=32,
                            shuffle=True,
                            drop_last=True,
                            collate_fn=to_tensor)
    return dataloader


def get_sample():
    for i, data in enumerate(get_dataloader()):
        return data


if __name__ == '__main__':
    x, y, seg = get_sample()
    print(x[:2], x.shape)
    print(y[:5], y.shape)
    print(seg[:2], seg.shape)
