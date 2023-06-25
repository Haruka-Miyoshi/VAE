import torch.nn as nn

"""Decoder"""
class Decoder(nn.Module):
    def __init__(self, zd):
        super(Decoder, self).__init__()

        # Decoder
        self.__decoder=nn.Sequential(
            # zd:潜在変数空間の次元数->128次元
            nn.Linear(zd, 128),
            # 活性化関数 0未満は、切り捨て
            nn.ReLU(True),
            # 128次元->256次元
            nn.Linear(128, 256),
            # 活性化関数 0未満は、切り捨て
            nn.ReLU(True),
            # 256次元->784次元
            nn.Linear(256, 28 * 28),
            # 活性化関数 負の値も考慮
            nn.Tanh()
        )

    """順伝播"""
    def forward(self, x):
        # 潜在変数空間から復元
        __x_hat=self.__decoder(x)
        return __x_hat