import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from segmentation import function as seg_F
from ditection import function as src_F


def scale_to_width(img, height=640):
    """
    Resize the image by specifying the width and keeping the image ratio fixed.
    Parameters
    ----------
    img: input image
        cv2 image
    height: Specifying the width
        int

    Returns
    -------
    dst: resized image
        cv2 image
    """
    h, w = img.shape[:2]
    width = round(w * (height / h))
    dst = cv2.resize(img, dsize=(width, height))

    return dst


def cv2pil(image):
    """ OpenCV型 -> PIL型 """
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


if __name__ == '__main__':
    # Initial values
    image_path = "data/test5.jpg"
    image_height_scale = 720

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'use device: {device}')

    image = cv2.imread(image_path)
    image = scale_to_width(image, image_height_scale)
    height, width = image.shape[:2]
    image = cv2pil(image)
    person_image = seg_F.make_clipped_person(image, height, width, device)

    person_image.save('test.png')
