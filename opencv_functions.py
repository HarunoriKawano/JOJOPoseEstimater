import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def scale_to_height(img, height=720):
    """
    Resize the image by specifying the height and keeping the image ratio fixed.
    Parameters
    ----------
    img: input image
        cv2 image
    height: Specifying the height
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


def scale_to_width(img, width=720):
    """
    Resize the image by specifying the width and keeping the image ratio fixed.
    Parameters
    ----------
    img: input image
        cv2 image
    width: Specifying the width
        int

    Returns
    -------
    dst: resized image
        cv2 image
    """
    h, w = img.shape[:2]
    height = round(h * (width / w))
    dst = cv2.resize(img, dsize=(width, height))

    return dst


def image_synthesis(target, back, top_x, top_y, alpha):
    top_x = int(top_x)
    top_y = int(top_y)
    height, width = target.shape[:2]
    if back.ndim == 4:
        back = cv2.cvtColor(back, cv2.COLOR_RGBA2RGB)
    if target.ndim == 3:
        target = cv2.cvtColor(target, cv2.COLOR_RGB2RGBA)

    mask = target[:, :, 3]
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = mask / 255 * alpha

    target = target[:, :, :3]
    back = back.astype(np.float64)
    target = target.astype(np.float64)

    back[top_y:height + top_y, top_x:width + top_x] *= 1 - mask
    back[top_y:height + top_y, top_x:width + top_x] += target * mask

    return back.astype(np.uint8)


def pil2cv(imgPIL):
    imgCV_RGB = np.array(imgPIL, dtype=np.uint8)
    imgCV_BGR = np.array(imgCV_RGB)[:, :, ::-1]
    return imgCV_BGR


def cv2pil(imgCV):
    imgCV_RGB = imgCV[:, :, ::-1]
    imgPIL = Image.fromarray(imgCV_RGB)
    return imgPIL


def cv2_putText(img, text, org, fontFace, fontScale, color):
    x, y = org
    b, g, r = color
    colorRGB = (r, g, b)
    imgPIL = cv2pil(img)
    draw = ImageDraw.Draw(imgPIL)
    fontPIL = ImageFont.truetype(font=fontFace, size=fontScale)
    w, h = draw.textsize(text, font=fontPIL)
    draw.text(xy=(x, y - h), text=text, fill=colorRGB, font=fontPIL)
    imgCV = pil2cv(imgPIL)
    return imgCV
