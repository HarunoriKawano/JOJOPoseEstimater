import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


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


def toRGBA(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    return image


def cv2_putText(text, back, org, font_path, font_size, color=(255, 255, 255), edge_color=(255, 255, 255)):
    x1, y1 = org
    r, g, b = color
    color = (b, g, r, 255)
    r, g, b = edge_color
    edge_color = (b, g, r, 255)
    font = ImageFont.truetype(font_path, font_size)

    image = Image.fromarray(back)
    draw_dummy = ImageDraw.Draw(image)  # 描画用のDraw関数をダミーで用意（テキストサイズ計測用）
    w, h = draw_dummy.textsize(text, font)

    if w == 0:
        return back
    img_bgd = Image.fromarray(back[y1:y1+h, x1:x1+w])
    draw = ImageDraw.Draw(img_bgd)

    draw.text((0, 0), text, font=font, fill=(255, 255, 255, 255))

    pad = int(h * 0.2)
    img_bgd = cv2.copyMakeBorder(np.array(img_bgd), pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)

    kernel = np.ones((5, 5), np.uint8)
    img_bgd = cv2.dilate(img_bgd, kernel, iterations=2)

    alpha = np.full((img_bgd.shape[0], img_bgd.shape[1]), 255, dtype=img_bgd.dtype)
    b, g, r = cv2.split(img_bgd)
    img_bgd = cv2.merge((b, g, r, alpha))

    color_lower = np.array([255, 255, 255, 255])
    color_upper = np.array([255, 255, 255, 255])

    mask = cv2.inRange(img_bgd, color_lower, color_upper)

    img_bgd[np.where((img_bgd[:, :, :3] == (255, 255, 255)).all(axis=2))] = edge_color

    img_bgd = np.array(img_bgd)
    bool = cv2.bitwise_and(img_bgd, img_bgd, mask=mask)

    bool = cv2.GaussianBlur(bool, (33, 33), 0)

    bool = Image.fromarray(bool)
    draw = ImageDraw.Draw(bool)
    draw.text((pad, pad), text, font=font, fill=color)
    bool = np.array(bool)

    w_bgd = img_bgd.shape[1]
    h_bgd = img_bgd.shape[0]
    x2 = x1 + w_bgd
    y2 = y1 + h_bgd

    image = np.array(image)

    image[y1:y2, x1:x2, :3] = image[y1:y2, x1:x2] * (1 - bool[:, :, 3:] / 255) + \
                            bool[:, :, :3] * (bool[:, :, 3:] / 255)

    image = np.array(image)
    return image


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False
