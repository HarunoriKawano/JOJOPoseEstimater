"""data loader programs"""
import itertools
from PIL import Image
import torch.utils.data as data
from pycocotools.coco import COCO

from data_augumentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor


def make_datapath_list(json_path, data_path):
    """

    Parameters
    ----------
    json_path: str
        path for json data file
    data_path: str
        path for image data file
    Returns
    -------
    train_img_list, train_anno_list, val_img_list, val_anno_list: tuple
        A list containing the path to the data
    """

    # get the coco objects
    train_json = "instances_train2017.json"
    val_json = "instances_val2017.json"
    coco_train = COCO(json_path + train_json)
    coco_val = COCO(json_path + val_json)

    # get image data
    train_person_ids = coco_train.getCatIds(catNms=['person'])
    val_person_ids = coco_val.getCatIds(catNms=['person'])

    train_img_ids = coco_train.getImgIds(catIds=train_person_ids)
    val_img_ids = coco_val.getImgIds(catIds=val_person_ids)

    train_file_names = list(itertools.chain.from_iterable([coco_train.loadImgs(target) for target in train_img_ids]))
    val_file_names = list(itertools.chain.from_iterable([coco_val.loadImgs(target) for target in val_img_ids]))

    train_img_data = [data_path + "train2017/" + target["file_name"] for target in train_file_names]
    val_img_data = [data_path + "val2017/" + target["file_name"] for target in val_file_names]

    # get annotation data
    train_anno_data = [data_path + "train_person_segmentation2017/" + target["file_name"] for target in train_file_names]
    val_anno_data = [data_path + "val_person_segmentation2017/" + target["file_name"] for target in val_file_names]

    return train_img_data, train_anno_data, val_img_data, val_anno_data


class DataTransform:
    """
    Attributes
    ----------
    input_size: int
        リサイズ先の画像の大きさ
    color_mean: (R, G, B)
        各色チャンネルの平均値
    color_std: (R, G, B)
        各色チャンネルの標準偏差
    """

    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose([
                Scale(scale=[0.5, 1.5]),  # 画像の拡大
                RandomRotation(angle=[-10, 10]),  # 回転
                RandomMirror(),
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ]),
            'val': Compose([
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        """

        Parameters
        ----------
        phase: 'train' or 'val'
        img: input image
        anno_class_img: annotation image

        Returns
        -------
        transformed image: torch.tensor(
        """
        return self.data_transform[phase](img, anno_class_img)


class VOCDataset(data.Dataset):
    """
    Class to create a VOC Dataset

    Attributes
    ----------
    img_list: list()
        A list containing the paths to the images
    anno_list: list()
        A list containing the paths to the annotation
    phase: 'train' or 'test'
        Set train or test
    transform: object
        preprocessing class instance
    """

    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        """return number of image"""
        return len(self.img_list)

    def __getitem__(self, index):
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        # load images
        img_file_path = self.img_list[index]
        img = Image.open(img_file_path)
        img = img.convert("RGB")

        # load annotation images
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)

        # preform pretreatment
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        anno_class_img = anno_class_img <= 128

        return img, anno_class_img
