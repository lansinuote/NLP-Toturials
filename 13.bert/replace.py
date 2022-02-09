import random

import data


def random_replace(x):
    # x = [b, 60]
    # 不影响原来的x
    x = x.clone()

    # 替换矩阵,形状和x一样,被替换过的位置是True,其他位置是False
    replace = x == -1

    # 遍历所有的字
    for i in range(len(x)):
        for j in range(len(x[i])):
            # 如果是符号就不操作了,只替换字
            if x[i, j] == data.zidian['<PAD>'] \
                    or x[i, j] == data.zidian['<SOS>'] \
                    or x[i, j] == data.zidian['<EOS>']:
                continue

            # 0.15的概率做操作
            if random.random() > 0.15:
                continue

            # 被操作过的位置标记下,这里的操作包括什么也不做
            replace[i, j] = True

            # 分概率做不同的操作
            p = random.random()

            # 0.7的概率替换为mask
            if p < 0.7:
                x[i, j] = data.zidian['<MASK>']

            # 0.15的概率不替换
            elif p < 0.85:
                pass

            # 0.15的概率替换成随机字
            else:
                # 随机一个不是符号的字
                rand_word = random.randint(0, len(data.zidian) - 1)
                while rand_word == data.zidian['<PAD>'] \
                        or rand_word == data.zidian['<SOS>'] \
                        or rand_word == data.zidian[
                            '<EOS>']:
                    rand_word = random.randint(0, len(data.zidian) - 1)
                x[i, j] = rand_word

    return x, replace


if __name__ == '__main__':
    x, y, seg = data.get_sample()

    replace_x, replace = random_replace(x)

    print(x[0])
    print(replace_x[0])
    print(replace[0])
