"""PSPNet network module"""
import torchinfo

from segmentation.feature_map_convolution import FeatureMap_convolution
from segmentation.residual_block_PSP import ResidualBlockPSP
from segmentation.pyramid_pooling import PyramidPooling
from segmentation.decorder_and_auxloss import *


class PSPNet(nn.Module):
    def __init__(self, n_classes):
        super(PSPNet, self).__init__()

        # Set parameters
        block_config = [3, 4, 6, 3]  # resnet50
        img_size = 475
        img_size_8 = 60

        # Sub-networks comprising the four modules
        self.feature_conv = FeatureMap_convolution()
        self.feature_res_1 = ResidualBlockPSP(
            n_blocks=block_config[0], in_channels=128, mid_channels=64,
            out_channels=256, stride=1, dilation=1
        )
        self.feature_res_2 = ResidualBlockPSP(
            n_blocks=block_config[1], in_channels=256, mid_channels=128,
            out_channels=512, stride=2, dilation=1
        )
        self.feature_dilated_res_1 = ResidualBlockPSP(
            n_blocks=block_config[2], in_channels=512, mid_channels=256,
            out_channels=1024, stride=1, dilation=2
        )
        self.feature_dilated_res_2 = ResidualBlockPSP(
            n_blocks=block_config[3], in_channels=1024, mid_channels=512,
            out_channels=2048, stride=1, dilation=4
        )

        self.pyramid_pooling = PyramidPooling(
            in_channels=2048, pool_sizes=[6, 3, 2, 1],
            height=img_size_8, width=img_size_8
        )
        self.decode_feature = DecodePSPFeature(
            height=img_size, width=img_size, n_classes=n_classes
        )
        self.aux = AuxiliaryPSPlayers(
            in_channels=1024, height=img_size, width=img_size, n_classes=n_classes
        )

    def forward(self, x):
        x = self.feature_conv(x)
        x = self.feature_res_1(x)
        x = self.feature_res_2(x)
        x = self.feature_dilated_res_1(x)
        output_aux = self.aux(x)

        x = self.feature_dilated_res_2(x)

        x = self.pyramid_pooling(x)
        output = self.decode_feature(x)

        return output, output_aux


class PSPLoss(nn.Module):
    """loss function class for PSPNet"""

    def __init__(self, aux_weight=0.4):
        super(PSPLoss, self).__init__()
        self.aux_weight = aux_weight

    def forward(self, outputs, targets):
        """
        calculating the loss function

        Parameters
        ----------
        outputs: PSPNet output
            (output=torch.Size([num_batch, 21, 475, 475]), output_aux=torch.Size([num_batch, 21, 475, 475]))

        targets: [num_batch, 475, 475]
            Annotation information for correct answers

        Returns
        -------
        loss: tensor
            loss value
        """

        loss = F.cross_entropy(outputs[0], targets, reduction='mean')
        loss_aux = F.cross_entropy(outputs[1], targets, reduction='mean')

        return loss+self.aux_weight*loss_aux


if __name__ == '__main__':
    net = PSPNet(21)
    batch_size = 8
    torchinfo.summary(net,
                      input_size=(batch_size, 3, 475, 475))
