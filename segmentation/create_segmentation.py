import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import itertools


def make_datapath_list(json_path):
    """

    Parameters
    ----------
    anno_path: str
        path for json data file
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

    train_img_data = [coco_train.loadImgs(target) for target in train_img_ids]
    val_img_data = [coco_val.loadImgs(target) for target in val_img_ids]

    # get annotation data
    train_anno_ids = [coco_train.getAnnIds(imgIds=target, catIds=train_person_ids, iscrowd=False) for target in
                      train_img_ids]
    val_anno_ids = [coco_val.getAnnIds(imgIds=target, catIds=val_person_ids, iscrowd=False) for target in val_img_ids]

    train_anno_data = [coco_train.loadAnns(target) for target in train_anno_ids]
    val_anno_data = [coco_val.loadAnns(target) for target in val_anno_ids]

    return list(itertools.chain.from_iterable(train_img_data)), list(itertools.chain.from_iterable(train_anno_data)), \
           list(itertools.chain.from_iterable(val_img_data)), list(itertools.chain.from_iterable(val_anno_data)), \
           coco_train, coco_val


if __name__ == '__main__':
    datapath = "D:\\LearningData/COCO2017/annotations/"
    save_path = "D:\\LearningData/COCO2017/"
    train_img, train_anno, val_img, val_anno, coco_train, coco_val = make_datapath_list(datapath)

    for coco, imgs, anns in zip((coco_train, coco_val), (train_img, val_img), (train_anno, val_anno)):
        folder_name = 'train_person_segmentation2017/' if coco is coco_train else 'val_person_segmentation2017/'
        if coco is coco_train:
            continue

        for i, img in enumerate(imgs):
            anno = [anns[i] for i in range(len(anns)) if anns[i]['image_id'] == img['id']]
            mask = np.zeros((img['height'], img['width']))
            for i in range(len(anno)):
                mask = np.maximum(coco.annToMask(anno[i]), mask)

            mask[mask != 0] = 255
            mask = Image.fromarray(mask)
            mask = mask.convert('L')
            save_file_path = save_path + folder_name + img['file_name']
            mask.save(save_file_path)
