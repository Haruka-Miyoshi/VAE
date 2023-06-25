import torch
import torch.nn as nn

"""VariationalZ"""
class VariationalZ(nn.Module):
    """コンストラクタ"""
    def __init__(self):
        super(VariationalZ, self).__init__()
        # デバイス設定 GPU or CPU
        self.__device="cuda" if torch.cuda.is_available() else "cpu"
    
    """潜在変数空間からサンプリング"""
    def sampling(self, mu, sigma):
        # ガウス分布を使っていないので注意
        epsilon=torch.randn(64, device=self.__device)
        return mu + epsilon * torch.exp(0.5 * sigma)
        