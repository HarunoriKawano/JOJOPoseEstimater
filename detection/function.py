import torch

from detection import models


def inference(image):
    vgg = models.make_vgg16()

    vgg_weights = torch.load('../weights/vgg16_reducedfc.pth')
    vgg.load_state_dict(vgg_weights)

    dense = models.make_dense(11)

    result = vgg(image)
    result = result.view(result.size(0), -1)
    out = dense(result)

    return out