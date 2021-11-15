import torch
from PIL import Image
import numpy as np
import cv2

from segmentation import PSPNet, DataTransform


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


def pil2cv(image):
    """ PIL型 -> OpenCV型 """
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]
    return new_image


def make_clipped_person(image, height, width, device):
    # Cut out the target.
    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)
    dummy_annotation = np.zeros((height, width))
    dummy_annotation = Image.fromarray(dummy_annotation)
    annotation_image = cv2pil(image)
    transformed_image, _ = DataTransform(475, color_mean, color_std)("val", annotation_image, dummy_annotation)

    psp_net = PSPNet(n_classes=2)
    psp_net.load_state_dict(torch.load("segmentation/weights/pspnet50_2_20.pth"))  # weight
    psp_net.to(device)
    psp_net.eval()
    dummy_image = torch.zeros(3, 475, 475)
    transformed_image = torch.stack((transformed_image, dummy_image), 0)
    transformed_image = transformed_image.to(device)
    _, annotation = psp_net(transformed_image)
    annotation = annotation[0]

    numpy_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    annotation = annotation.to('cpu').detach().numpy()
    annotation = np.argmax(annotation, axis=0)
    annotation = annotation * 255
    annotation = Image.fromarray(np.uint8(annotation)).convert('L')
    annotation = annotation.resize((width, height), Image.NEAREST)
    annotation = pil2cv(annotation)
    contours = cv2.findContours(
        annotation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    try:
        max_cnt = max(contours, key=lambda x: cv2.contourArea(x))
    except ValueError:
        print('人を検出できませんでした。')

    annotation = np.zeros_like(annotation)
    cv2.drawContours(annotation, [max_cnt], -1, color=255, thickness=-1)

    index = np.where(annotation == 0)
    dis_index = np.where(annotation != 0)
    numpy_image[index] = 0
    person_image = numpy_image[dis_index[0].min(): dis_index[0].max(), dis_index[1].min(): dis_index[1].max(), :]

    return person_image
