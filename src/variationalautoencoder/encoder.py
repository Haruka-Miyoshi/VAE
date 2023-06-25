import torch.nn as nn

"""Encoder"""
class Encoder(nn.Module):
    """コンストラクタ"""
    def __init__(self, zd):
        super(Encoder, self).__init__()
        # Encoder
        self.__encoder=nn.Sequential(
            # 784次元->256次元
            nn.Linear(28 * 28, 256),
            # 活性化関数 0未満は、切り捨て
            nn.ReLU(True),
            # 256次元->128次元
            nn.Linear(256, 128),
            # 活性化関数 0未満は、切り捨て
            nn.ReLU(True)
        )
        # 潜在変数空間における平均パラメータ層
        self.__mu=nn.Linear(128, zd)
        # 潜在変数空間における分散パラメータ層
        self.__sigma=nn.Linear(128, zd)
    
    """順伝播"""
    def forward(self,x):
        # 潜在変数空間へ圧縮
        x=self.__encoder(x)
        # 平均パラメータ層
        mu=self.__mu(x)
        # 分散パラメータ層
        sigma=self.__sigma(x)
        return mu, sigma