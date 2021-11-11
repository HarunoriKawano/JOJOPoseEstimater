#  image_loader.py
#  1.画像のロード, 前処理, データのかさ増し, ミニバッチ作成

import numpy as np
import pandas as pd
from PIL import Image
import os, glob, random

import torch
import torch.utils.data as data


LABELS = ["Buccellati", "Dio", "Giorno", "Highway-Star", "Jo-suke", "Jo-taro",
            "Kakyoin", "Kira", "Kishibe", "Polnareff", "Trish"]


def data_loder(root_path, phase):
    """ path以下の画像を読み込む

    Parameters:
        root_path(list): ルートパス
        phase(str): 'train'または'test', 訓練か検証を指定

    Returns:
        img_path_list(Tensor): imageのpathを格納したリスト
            (label数(11)*各labelごとのimage, )
        labels(Tensor): labelを格納したリスト 
            (label数(11)*各labelごとのimage, )
    """

    img_path_list = []
    labels = []

    root_dir = root_path + "/images/" + phase

    for i,label in enumerate(LABELS):
        files = glob.glob(root_dir + "/" + label + "/*.png")
        #  random.shuffle(files)
        #  各ファイルを処理
        for f in files:
            img_path_list.append(f)
            labels.append(i)

    return img_path_list, torch.tensor(labels)


class PreprocessJOJO(data.Dataset):
    """ PytorchのDatasetクラスを継承
        前処理をした後,以下のデータを返す

        ・前処理後のイメージ[R,G,B](Tensor)
        ・ラベル(Tensor)
        ・イメージの高さ,幅(int)
    """

    def __init__(self, img_path_list, labels, phase, im_rows, im_cols, transform=None):
        """
        Parameters:
            img_path_list(list): imageのpathを格納したリスト
            labels(Tensor): labelを格納したリスト
            phase(str): 'train'または'test', 訓練か検証を指定
            im_rows(int): imageの縦幅
            im_cols(int): imageの横幅
            transform(object): 前処理クラスDataTransform(ある場合は指定)
        """
        self.img_path_list = img_path_list
        self.labels = labels
        self.phase = phase
        self.im_rows = im_rows
        self.im_cols = im_cols
        self.transform = transform

    def __len__(self):
        """imageの数を返す
        """
        return len(self.img_path_list)

    def __getitem__(self, index):
        """ データの数だけイテレート
            前処理後のimage及びlabelを取得する

        Parameters:
            index(int): imageのindex

        Returns:
            img(Tensor): 前処理後のimage(im_rows, im_cols, 3)
            label(str): 正解label
        """
        img, label, _, _ = self.pull_item(index)
        return img, label

    def pull_item(self, index):
        """ 前処理後, テンソル形式のimage, label
            imageの高さ(h), 幅(w)

        Parameter:
            index(int): imageのindex

        Returns:
            img(Tensor): 前処理後のimage(im_rows, im_cols, 3)
            label(str): 正解label
            height(int): imageの縦幅
            width(int): imageの横幅
        """

        # 画像ファイルを読む
        img = Image.open(self.img_path_list[index])
        img = img.convert("RGB") #  色空間をRGBに
        img = img.resize((self.im_rows, self.im_cols)) # サイズ変更
        img = np.asarray(img)
        img = torch.tensor(img)
        img = img/255#  正規化
        img = img.permute(2,0,1).float()
        height, width, _ = img.shape
        label = self.labels[index].long()

        #  image前処理
        #  img, label = self.transform(img, self.phase)

        return img, label, height, width


if __name__ == "__main__":

    im_rows = 256
    im_cols = 256

    #  Dataを取得
    train_img_path_list, train_labels = data_loder("train")
    valid_img_path_list, valid_labels = data_loder("valid")

    #  Datasetを作成
    tr_data = PreprocessJOJO(train_img_path_list, train_labels, "train", im_rows, im_cols)
    val_data = PreprocessJOJO(valid_img_path_list, valid_labels, "valid", im_rows, im_cols)
    print('訓練データのサイズ: ', tr_data.__len__())
    print('検証データのサイズ: ', val_data.__len__())

    #  DataLorderを作成
    batch_size = 12
    tr_batch = data.DataLoader(
        tr_data,                #  訓練用data
        batch_size = batch_size,#  ミニバッチのサイズ
        shuffle = True,         #  シャッフルして抽出
        )
    val_batch = data.DataLoader(
        val_data,               #  検証用data
        batch_size = batch_size,#  ミニバッチのサイズ
        shuffle = False,        #  シャッフルはせずに抽出
        )
    print('訓練データのミニバッチの個数: ', tr_batch.__len__())
    print('検証データのミニバッチの個数: ', val_batch.__len__())

    #  DataLorderをdictにまとめる
    dataloders_list = {"train":tr_batch, "valid":val_batch}

    #  訓練用のDataLorderをイテレーターに変換
    batch_iterator = iter(dataloders_list["train"])

    #  最初のミニバッチを取り出す
    images, labels = next(batch_iterator)
    print('ミニバッチのイメージの形状: ',images.size())
    print('ミニバッチのラベルの形状: ',len(labels))
    print('labels[0]の形状: ',labels[0].size())
