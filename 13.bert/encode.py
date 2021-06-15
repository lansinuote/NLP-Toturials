import torch.nn as nn

import mask

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 定义编码层
        encoder_layer = nn.TransformerEncoderLayer(d_model=256,
                                                   nhead=4,
                                                   dim_feedforward=256,
                                                   dropout=0.2,
                                                   activation='relu')

        # 定义规范化层
        norm = nn.LayerNorm(normalized_shape=256, elementwise_affine=True)

        # 定义编码器
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=3,
                                             norm=norm)

    def forward(self, x, mask_x):
        # 转换成torch要求的格式
        # [b, 60, 256] -> [60, b, 256]
        x = x.permute(1, 0, 2)

        # 编码层计算
        # [60, b, 256] -> [60, b, 256]
        out = self.encoder(src=x, mask=mask.attn_mask, src_key_padding_mask=mask_x)

        # 转换回自己的格式
        # [60, b, 256] -> [b, 60, 256]
        out = out.permute(1, 0, 2)

        return out
