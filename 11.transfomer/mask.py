import torch

import data


def mask_x(x):
    # b句话,每句话11个词,这里是还没embed的
    # x = [b, 11]
    # 判断每个词是不是<PAD>
    mask = x == data.zidian['<PAD>']

    # [b, 11] -> [b, 1, 1, 11]
    mask = mask.reshape(-1, 1, 1, 11)

    # [b, 1, 1, 11] -> [b, 1, 11, 11]
    mask = mask.expand(-1, 1, 11, 11)

    return mask


def mask_y(y):
    # b句话,每句话11个词,这里是还没embed的
    # y = [b, 11]

    b = y.shape[0]

    # b句话,11*11的矩阵表示每个词对其他词是否可见
    mask = torch.zeros(b, 11, 11)

    # 遍历b句话
    for bi in range(b):
        # 遍历11个词
        for i in range(11):
            # 如果词是pad,则这个词完全不可见
            if y[bi, i] == data.zidian['<PAD>']:
                mask[bi, :, i] = 1
                continue

            # 这个词之前的词都可见,之后的词不可见
            col = [1] * i + [0] * 11
            col = col[:11]
            mask[bi, :, i] = torch.LongTensor(col)

    # 转布尔型,增加一个维度,便于后续的计算
    mask = (mask == 1).unsqueeze(dim=1)

    return mask


if __name__ == '__main__':
    print(mask_x(data.get_sample()[0]).shape)
    print(mask_y(data.get_sample()[1][:, :-1]).shape)
