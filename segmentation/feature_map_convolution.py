from torch import nn

from segmentation.conv2D_batch_norm import conv2DBatchNormRelu


class FeatureMap_convolution(nn.Module):
    def __init__(self):
        super(FeatureMap_convolution, self).__init__()

        # convolution layer 1
        self.cbnr_1 = conv2DBatchNormRelu(
            in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, bias=False
        )

        # convolution layer 2
        self.cbnr_2 = conv2DBatchNormRelu(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False
        )

        # convolution layer 3
        self.cbnr_3 = conv2DBatchNormRelu(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False
        )

        # MaxPooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr_1(x)
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        outputs = self.maxpool(x)
        return outputs