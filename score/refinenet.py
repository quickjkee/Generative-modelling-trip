import torch.nn as nn
from layers import CondResBlock, RefineBlock


class RefineNet(nn.Module):
    def __init__(self, in_channels, channels, n_classes):
        """
        :param in_channels: Number of channels in input image
        :param channels:  Channels for networks
        """
        super(RefineNet, self).__init__()

        self.inp_conv = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.ELU()

        self.res_block1 = CondResBlock(n_classes=n_classes,
                                       in_channels=channels,
                                       out_channels=channels * 2)
        self.res_block2 = CondResBlock(n_classes=n_classes,
                                       in_channels=channels * 2,
                                       out_channels=channels * 4)
        self.res_block3 = CondResBlock(n_classes=n_classes,
                                       in_channels=channels * 4,
                                       out_channels=channels * 6)
        self.res_block4 = CondResBlock(n_classes=n_classes,
                                       in_channels=channels * 6,
                                       out_channels=channels * 8)

        self.refnet1 = RefineBlock(in_planes=[channels * 2, channels * 2],
                                   out_features=channels)
        self.refnet2 = RefineBlock(in_planes=[channels * 4, channels * 4],
                                   out_features=channels * 2)
        self.refnet3 = RefineBlock(in_planes=[channels * 6, channels * 6],
                                   out_features=channels * 4)
        self.refnet4 = RefineBlock(in_planes=[channels * 8],
                                   out_features=channels * 6)

    def forward(self, x, y):
        """
        :param x: (Tensor), [b_size x C x H x W], input image
        :param y: (Tensor), [b_size x 1], labels of noise
        :return: (Tensor), [b_size x C x H x W], output score
        """

        # (B x C x H x W) -> (B x C x H x W)
        out = self.inp_conv(x)

        # (B x C x H x W) -> (B x C*2 x H/2 x W/2)
        layer1 = self.res_block1(out, y)

        # (B x C*2 x H/2 x W/2) -> (B x C*4 x H/4 x W/4)
        layer2 = self.res_block2(layer1, y)

        # (B x C*4 x H/4 x W/4) -> (B x C*6 x H/8 x W/8)
        layer3 = self.res_block3(layer2, y)

        # (B x C*6 x H/8 x W/8) -> (B x C*8 x H/16 x W/16)
        layer4 = self.res_block4(layer3, y)

        # -> (B x C*6 x H/8 x W/8)
        ref4 = self.refnet4([layer4])

        # -> (B x C*4 x H/4 x W/4)
        ref3 = self.refnet3([layer3, ref4])

        # -> (B x C*2 x H/2 x W/2)
        ref2 = self.refnet2([layer2, ref3])

        # -> (B x C x H x W)
        out = self.refnet1([layer1, ref2])

        out = self.out_conv(self.act(out))

        return out


