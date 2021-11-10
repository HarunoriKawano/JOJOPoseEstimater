#  models.py
#  2.モデルのクラスを記述

import torch
import torch.nn as nn


def make_vgg16():
    """
    Returns:
        (nn.ModuleList): vggのモジュールのリスト
    """
    layers = []  # モジュールを格納するリスト
    input_channels = 3  # RGBで3チャネル

    # vggに配置する畳み込み層のフィルター数, 'M''MC'はプーリング層を示す
    cfg = [64, 64, 'M',  # vgg1
           128, 128, 'M',  # vgg2
           256, 256, 256, 'MC',  # vgg3
           512, 512, 512, 'M',  # vgg4
           512, 512, 512  # vgg5
           ]

    # vgg1～vgg5の畳み込み層までを生成
    for v in cfg:
        # vgg1、vgg2、vgg4のプーリング層
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2,  # ウィンドウサイズ2×2
                                    stride=2)]  # ストライド2
        # vgg3のプーリング層
        elif v == 'MC':
            # vgg3のプーリングで(75, 75)の特徴量マップを半分のサイズにする際に、
            # ceil_modeをTrueにすることで75/2=37.5を切り上げて38にする
            layers += [nn.MaxPool2d(kernel_size=2,  # ウィンドウサイズ2×2
                                    stride=2,  # ストライド2
                                    ceil_mode=True)]
        # vgg1～vgg5の畳み込み層
        else:
            conv2d = nn.Conv2d(input_channels,  # 入力時のチャネル数
                               v,  # 出力時のチャネル数(フィルター数)
                               kernel_size=3,  # フィルターサイズ3×3
                               padding=1)  # パディングのサイズは1

            # 畳み込み層に活性化関数ReLUをセットしてlayersに追加
            layers += [conv2d, nn.ReLU(inplace=True)]
            # チャネル数を出力時のチャネル数(フィルター数)に置き換える
            input_channels = v

    # vgg5のプーリング層
    pool5 = nn.MaxPool2d(kernel_size=3,  # ウィンドウサイズ3×3
                         stride=1,  # ストライド1
                         padding=1)  # パディングのサイズは1
    # vgg6の畳み込み層1
    conv6 = nn.Conv2d(512,  # 入力時のチャネル数
                      1024,  # 出力時のチャネル数(フィルター数)
                      kernel_size=3,  # フィルターサイズ3×3
                      padding=6,  # パディングのサイズは6
                      dilation=6)  # 畳み込みのポイント間の間隔を6にする
    # vgg6の畳み込み層2
    conv7 = nn.Conv2d(1024,  # 入力時のチャネル数
                      1024,  # 出力時のチャネル数(フィルター数)
                      kernel_size=1)  # フィルターサイズ1×1
    # vgg5のプーリング層、vgg6の畳み込み層1と畳み込み層2をlayersに追加
    layers += [pool5,
               conv6, nn.ReLU(inplace=True),  # 畳み込みの活性化はReLU
               conv7, nn.ReLU(inplace=True)]  # 畳み込みの活性化はReLU

    # リストlayersをnn.ModuleListに格納してReturnする
    return nn.ModuleList(layers)


def make_dense(classes_num):
    """
    Attributes:
        classes_num(int): クラスの数
    Returns:
        (nn.ModuleList): denseのモジュールのリスト
    """
    layers = []  # モジュールを格納するリスト
    input_channels = 1024 * 16 * 16  # vgg16より
    classes_num = classes_num

    linear1 = nn.Linear(input_channels, 512)
    linear2 = nn.Linear(512, 32)
    linear3 = nn.Linear(32, classes_num)

    layers += [nn.Flatten(),
               linear1, nn.ReLU(inplace=True), nn.Dropout(0.4),
               linear2, nn.ReLU(inplace=True), nn.Dropout(0.4),
               linear3]
    #  推論の際にはSoftmax関数を追加

    return nn.ModuleList(layers)


class JOJO_classifier(nn.Module):
    """ 画像からキャラクターを推測するクラス

    Attributes:
        phase(str): 'train'または'test'
        classes_num(int): クラスの数
        vgg(object): vggネットワーク
        dense(object): denseネットワーク
    """

    def __init__(self, phase, classes_num):
        """ インスタンス変数の初期化
        """
        super(JOJO_classifier, self).__init__()
        self.phase = phase
        self.classes_num = classes_num

        #  ネットワーク生成
        self.vgg = make_vgg16()
        self.dense = make_dense(self.classes_num)

    def forward(self, x):
        """ モデルの順伝播を行う
        Parameters:
            x: 画像を格納した4階テンソル
                (バッチサイズ, im_rows, im_cols, 3)
        Returns:
            out: 各画像, 各クラスに対する確信度
                (バッチサイズ, classes_num,)
        """
        for f in self.vgg:
            x = f(x)
        #  x = x.view(x.size(0),-1)
        for f in self.dense:
            x = f(x)
        return x


if __name__ == "__main__":
    vgg = make_vgg16()
    print(vgg)

    vgg_weights = torch.load('../weights/vgg16_reducedfc.pth')
    vgg.load_state_dict(vgg_weights)
    print("[model vgg] weights is applied.")

    dense = make_dense(11)
    print(dense)

    """
    x = vgg(x)
    x = x.view(x.size(0),-1)
    out = dense(x)
    """
