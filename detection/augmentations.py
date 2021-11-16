
import cv2
import numpy as np
from numpy import random

'''
データの拡張処理 実行
'''
class Compose(object):
    def __init__(self, transforms):
        '''
        Args:
            transforms (List[Transform]): 変換処理のリスト
        '''
        self.transforms = transforms

    def __call__(self, img, label=None, cast_integer=True):
        for t in self.transforms:
            img, label = t(img, label)

        if cast_integer:

            img = (img - img.min()) / (img.max() - img.min())

            """
            img = 255 * (img - img.min()) / (img.max() - img.min())
            img = img.astype(np.uint8)
            """
        return img, label


'''
ピクセルデータのint型をfloat32に変換
'''
class ConvertFromInts(object):
    def __call__(self, image, label=None):
        return image.astype(np.float32), label


'''
輝度をランダムに変化させる
'''
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, label=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, label


'''
コントラストをランダムに変化
'''
class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, label=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, label


'''
BGRとHSVを相互変換 (彩度,色相をいじるため)
'''
class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, label=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, label


'''
彩度をランダムに変化
'''
class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, label=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image, label


'''
色相をランダムに変化
'''
class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, label=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, label



'''
測光に歪みを加える
'''
class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, label=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, label


'''
色チャネルの並び順を変えるクラス(上記の関数で使用)
'''
class SwapChannels(object):
    def __init__(self, swaps):
        '''
        Args:
            swaps (int triple): final order of channels
                eg: (2, 1, 0)
        '''
        self.swaps = swaps

    def __call__(self, image):
        '''
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        '''
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image

'''
上記の内, 複数の関数を実行 (コントラストを変えるタイミングが前後で二種類)
'''
class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            # コントラスト(BGRに適用)
            RandomContrast(),# =========================
            # カラーモデルをHSVにコンバート
            ConvertColor(transform='HSV'),
            # 彩度の変化(HSVに適用)
            RandomSaturation(),
            # 色相の変化(HSVに適用)
            RandomHue(),
            # カラーモデルをHSVからBGRにコンバート
            ConvertColor(current='HSV', transform='BGR'),
            # コントラストの変化(BGRに適用)
            RandomContrast()# =========================
        ]
        # 輝度
        self.rand_brightness = RandomBrightness()# =========================
        # 測光の歪み
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, label):
        im = image.copy()
        # 明るさの変化
        im, label = self.rand_brightness(im, label)
        # 彩度、色相、コントラストの適用は上限と下限の間でランダムに
        # 歪みオフセットを選択することにより、確率0.5で適用
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        # 彩度、色相、コントラストの適用
        im, label = distort(im, label)
        return self.rand_light_noise(im, label)


'''
イメージをランダムに拡大
'''
class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, label):
        if random.randint(2):
            return image, label

        height, width, depth = image.shape
        ratio = random.uniform(1, 1.2)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        return image, label


'''
イメージの左右をランダムに反転
'''
class RandomMirror(object):
    def __call__(self, image, label):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
        return image, label


'''
イメージのサイズをinput_sizeにリサイズするクラス
'''
class Resize(object):
    def __init__(self, im_rows, im_cols):
        self.im_rows = im_rows
        self.im_cols = im_cols

    def __call__(self, image, label=None):
        image = cv2.resize(image,(self.im_rows, self.im_cols))
        return image, label


'''
色情報(RGB値）から平均値を引き算するクラス
'''
class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, label=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), label






