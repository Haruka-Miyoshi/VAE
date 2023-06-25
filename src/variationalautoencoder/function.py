import torch
from tqdm import tqdm
import os
import numpy as np
import torch.nn  as nn
from .model import Model

"""VAE"""
class VAE(object):
    """コンストラクタ"""
    def __init__(self, zd=64, mode=False, model_path=''):
        # デバイス設定 GPU or CPU
        self.__device="cuda" if torch.cuda.is_available() else "cpu"
        # モデル定義
        self.__model=Model(zd=zd).to(self.__device)

        if mode:
            # 学習済みモデル読み込み
            self.__model.load_state_dict(torch.load(model_path))
            self.__model.eval()

        # 学習係数
        self.__lr=1e-3
        # 損失関数:最小二乗法
        self.__loss_func=nn.MSELoss()
        # 最適化アルゴリズム:Adam
        self.__opt=torch.optim.Adam(self.__model.parameters(), lr=self.__lr)

        # save file path
        self.FILE_PATH=os.path.join('./model')

        # フォルダを生成
        if not os.path.exists(self.FILE_PATH):
            os.mkdir(self.FILE_PATH)

        # 損失値格納用変数
        self.__loss_history=[]
        
    """update:学習"""
    def update(self, data, mode=False):
        data=tqdm(data)
        # パラメータ計算
        for batch, (X, y) in enumerate(data):
            # 28*28を784次元に変換
            X=X.reshape(784)
            # device調整
            X=X.to(self.__device)
            # 学習用データXをAutoEncoderモデルに入力 -> 計算結果 出力Y
            X_hat=self.__model(X)

            # 損失計算(ラベルYと予測Yとの最小二乗法による損失計算)
            loss=self.__loss_func(X_hat, X)

            # 誤差逆伝播を計算
            # 勾配値を0にする
            self.__opt.zero_grad()
            # 逆伝播を計算
            loss.backward()
            # 勾配を計算
            self.__opt.step()
            
            loss=loss.item()
            # 損失を格納
            self.__loss_history.append(loss)

        # 損失保存
        if mode:
            """汎用的な保存方法を検討中"""
            # ファイル path
            LOSS_SAVE=os.path.join(self.FILE_PATH, 'loss.txt')
            # 損失結果 保存
            np.savetxt(LOSS_SAVE, self.__loss_history)
            # パラメータ保存
            PARAM_SAVE=os.path.join(self.FILE_PATH, 'parameter.pth')
            # 学習したパラメータを保存
            torch.save(self.__model.state_dict(), PARAM_SAVE)
        
    """test_loss:テストデータを使って損失計算"""
    def test_loss(self, data, mode=False):
        data=tqdm(data)
        # 勾配なし
        with torch.no_grad():
            # 損失
            loss=0
            # パラメータ計算
            for batch, (X, y) in enumerate(data):
                # 28*28を784次元
                X=X.reshape(784)
                # device調整
                X=X.to(self.__device)
                # 生成
                X_hat=self.__model(X)
                # 損失計算
                loss+=self.__loss_func(X_hat, X).item()
                    
        print("\n ====================== \n")
        print(f"loss:{loss}")
        print("\n ====================== \n")

                # 損失保存
        if mode:
            """汎用的な保存方法を検討中"""
            # ファイル path
            LOSS_SAVE=os.path.join(self.FILE_PATH, 'testloss.txt')
            # 損失結果 保存
            np.savetxt(LOSS_SAVE, [loss])

        return loss
    
    """generated:生成"""
    def generated(self, X):
        X=X.reshape(784)
        X=X.to(self.__device)
        # 生成
        X_hat=self.__model(X)
        X_hat=X_hat.reshape((28,28))
        return X_hat