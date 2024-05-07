import torch
from torch import nn
from torch.nn import functional as F
from .Decoder import Decoder
from .Encoder import Encoder

"""Model"""
class Model(nn.Module):
    """__init__"""
    def __init__(self, x_dim:int, h_dim:int, z_dim:int, mode=True):
        super(Model, self).__init__()
        self.x_dim = x_dim # 入力変数次元数
        self.h_dim = h_dim # 隠れ変数次元数
        self.z_dim = z_dim # 潜在変数次元数

        self.mode = mode # 学習モード

        self.encoder = Encoder(self.x_dim, self.h_dim, self.z_dim) # Encoder Module
        self.decoder = Decoder(self.x_dim, self.h_dim, self.z_dim) # Decoder Module

    """reparameterize"""
    def reparameterize(self, mu, logvar, mode):
        if mode:
            s = torch.exp(0.5 * logvar) # 標準偏差
            e = torch.rand_like(s) # 誤差e
            return e.mul(s).add_(mu) # e * std + mu
        else:
            return mu # mu
        
    """forward"""
    def forward(self, x, mode=True):
        mu, logvar = self.encoder(x) # mu, logvarを計算
        z = self.reparameterize(mu, logvar, mode) # reparameterize
        x_hat = self.decoder(z) # x_hatを計算
        return mu, logvar, z, x_hat