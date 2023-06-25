import torch
import torch.nn as nn
from .encoder import Encoder
from .variational_z import VariationalZ
from .decoder import Decoder

"""VAE"""
class Model(nn.Module):
    def __init__(self, zd):
        super(Model, self).__init__()
        # Encoder
        self.__encoder=Encoder(zd)
        # Variational Z
        self.__z=VariationalZ()
        # Decoder
        self.__decoder=Decoder(zd)

    def forward(self, x):
        # データを潜在変数空間へ圧縮
        __mu, __sigma=self.__encoder(x)
        # 潜在変数空間からサンプリング
        __z=self.__z.sampling(__mu, __sigma)
        # KLダイバージェンス
        kl=0.5 * torch.sum(1 + __sigma - __mu**2 - torch.exp(__sigma))
        # 潜在変数空間からデータを復元
        __xhat=self.__decoder(__z)
        return __xhat