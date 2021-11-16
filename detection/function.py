import cv2
import torch
import torch.nn as nn
from torchvision import models
import numpy as np

from detection.image_loader import DataTransform


def inference(image, device):
    #   モデルのインスタンス作成
    net = models.vgg16(pretrained=False)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=11)
    #  vggモデルの学習済みの重みを適用
    net_weights = torch.load('detection/weights/jojo_weights15.pth', map_location=torch.device(device))
    net.load_state_dict(net_weights)

    #  image 前処理
    # DataTransformで前処理を実施
    im_rows = 300
    im_cols = 300
    color_mean = (132, 140, 144)  # BGR
    transform = DataTransform(im_rows, im_cols, color_mean)
    image, label = transform(image, 'valid', 0)
    image = torch.from_numpy(image[:, :, (2, 1, 0)]).permute(2, 0, 1)
    image = torch.unsqueeze(image, 0)

    output = net(image)

    return output.data.max(1)[1].item()
