import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

zidian = {
    '<PAD>': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '0': 10,
    'Jan': 11,
    'Feb': 12,
    'Mar': 13,
    'Apr': 14,
    'May': 15,
    'Jun': 16,
    'Jul': 17,
    'Aug': 18,
    'Sep': 19,
    'Oct': 20,
    'Nov': 21,
    'Dec': 22,
    '-': 23,
    '/': 24,
    '<SOS>': 25,
    '<EOS>': 26,
}


class DateDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 2000

    def __getitem__(self, index):
        # 随机生成一个日期
        date = np.random.randint(143835585, 2043835585)
        date = datetime.datetime.fromtimestamp(date)

        # 格式化成两种格式
        # 05-06-15
        # 15/Jun/2005
        date_cn = date.strftime("%y-%m-%d")
        date_en = date.strftime("%d/%b/%Y")

        # 中文的就是简单的拿字典编码就行了
        # 补齐到和英文同样的长度
        date_cn_code = [zidian[v] for v in date_cn] + [zidian['<PAD>']] * 3

        # 英文的,首先要在收尾加上标志位,然后用字典编码
        date_en_code = []
        date_en_code += [zidian['<SOS>']]
        date_en_code += [zidian[v] for v in date_en[:3]]
        date_en_code += [zidian[date_en[3:6]]]
        date_en_code += [zidian[v] for v in date_en[6:]]
        date_en_code += [zidian['<EOS>']]
        date_en_code += [zidian['<PAD>']]

        return torch.LongTensor(date_cn_code), torch.LongTensor(date_en_code)


def get_dataloader():
    dataloader = DataLoader(dataset=DateDataset(),
                            batch_size=100,
                            shuffle=True,
                            drop_last=True)
    return dataloader


def get_sample():
    # 遍历数据
    for i, data in enumerate(get_dataloader()):
        return data


# 数字化的句子转字符串
def seq_to_str(seq):
    # 构造反转的字典
    reverse_zidian = {}
    for k, v in zidian.items():
        reverse_zidian[v] = k

    seq = seq.detach().numpy()
    return ''.join([reverse_zidian[idx] for idx in seq])


if __name__ == '__main__':
    x, y = get_sample()
    print(x[:5], x.shape)
    print(y[:5], y.shape)
