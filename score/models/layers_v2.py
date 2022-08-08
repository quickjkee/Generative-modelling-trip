import torch.nn as nn
import torch


##############################
#
# Residual Network
#
##############################

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=None, resize=False):
        """
        :param in_channels: (Int), number of input channels
        :param out_channels: (Int), number of output channels
        """
        super(ResBlock, self).__init__()

        self.act = nn.ELU()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if not resize:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
            self.shortcut = None

        else:
            self.shortcut = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.InstanceNorm2d(out_channels)
            ])
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        """
        :param x: (Tensor), [b_size x C x H x W], input image
        :param y: (Tensor), [b_size x 1], labels of noise
        :return: (Tensor), [b_size x C' x H/2 x W/2], output image
        """
        # (B x C x W x H) -> (B x C x W/2 x H/2)
        if self.shortcut:
            shortcut = self.shortcut[0](x)
            shortcut = self.shortcut[1](shortcut)
        else:
            shortcut = x

        # (B x C x W x H) -> (B x C x W x H)
        out = self.act(self.norm1(self.conv1(x)))

        # (B x C x W x H) -> (B x C x W/2 x H/2)
        out = self.act(self.norm2(self.conv2(out)))
        out = out + shortcut

        return self.act(out)


class InstanceNormPlus(nn.Module):
    def __init__(self, n_features, bias=True):
        super().__init__()
        self.n_features = n_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(n_features, affine=False, track_running_stats=False)
        self.alpha = nn.Parameter(torch.zeros(n_features))
        self.gamma = nn.Parameter(torch.zeros(n_features))
        self.alpha.data.normal_(1, 0.02)
        self.gamma.data.normal_(1, 0.02)
        if bias:
            self.beta = nn.Parameter(torch.zeros(n_features))

    def forward(self, x, y):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:

            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.n_features, 1, 1) * h + self.beta.view(-1, self.n_features, 1, 1)
        else:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.n_features, 1, 1) * h
        return out


##############################
#
# RefineNET
#
##############################


class RefineBlock(nn.Module):
    def __init__(self, in_planes, out_features):
        """
        :param in_planes: (List), list of channels of input images
        :param out_features: (Int), number of channels for output image
        """
        super(RefineBlock, self).__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = len(in_planes)
        self.in_planes = in_planes

        self.adap_conv = self._make_adaptive()
        self.mrf = MRFBlock(in_planes=in_planes,
                            out_features=out_features)
        self.crp = CRPBlock(features=out_features,
                            n_stages=2)
        self.out_conv = RCUBlock(features=out_features)

    def _make_adaptive(self):
        out = nn.ModuleList()

        for i in range(self.n_blocks):
            out.append(nn.Sequential(
                RCUBlock(features=self.in_planes[i]),
                RCUBlock(features=self.in_planes[i])
            ))

        return out

    def forward(self, xs):
        """
        :param xs: (List(Tensor)), input images
        :return: (Tensor), [b_size x C x W x H], output image
        """
        assert isinstance(xs, tuple) or isinstance(xs, list)

        out_adap = []
        for i, x in enumerate(xs):
            out_adap.append(self.adap_conv[i](x))

        out = self.mrf(out_adap)
        out = self.crp(out)
        out = self.out_conv(out)

        return out


class RCUBlock(nn.Module):
    def __init__(self, features):
        super(RCUBlock, self).__init__()

        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(features,
                               features,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(features,
                               features,
                               kernel_size=3,
                               stride=1,
                               padding=1)

    def forward(self, x):
        # (B x C x W x H) -> (B x C x W x H)
        out = self.conv1(self.act(x))
        out = self.conv2(self.act(out))

        return out + x


class MRFBlock(nn.Module):
    def __init__(self, in_planes, out_features):
        super(MRFBlock, self).__init__()

        self.in_planes = in_planes
        self.out_features = out_features

        self.model = self._make_model()

    def _make_model(self):
        out = nn.ModuleList()

        for i in range(len(self.in_planes)):
            out.append(nn.Sequential(
                nn.Conv2d(self.in_planes[i],
                          self.out_features,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.Upsample(scale_factor=2,
                            mode='bilinear')
            ))

        return out

    def forward(self, xs):
        assert isinstance(xs, tuple) or isinstance(xs, list)

        out = []
        for i, x in enumerate(xs):
            # (B x C x H x W) -> (B x C' x H*2 x W*2)
            out.append(self.model[i](x))

        return torch.stack(out, dim=0).sum(dim=0)


class CRPBlock(nn.Module):
    def __init__(self, features, n_stages=2):
        super(CRPBlock, self).__init__()

        self.n_stages = n_stages
        self.features = features

        self.act = nn.ELU()
        self.convs = self._make_convs()
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def _make_convs(self):
        convs = nn.ModuleList()

        for _ in range(self.n_stages):
            convs.append(nn.Sequential(nn.Conv2d(self.features,
                                                 self.features,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1)))

        return convs

    def forward(self, x):
        x = self.act(x)
        path = x

        for i in range(self.n_stages):
            # (B x C x H x W) -> (B x C x H x W)
            path = self.pool(path)
            path = self.convs[i](path)
            x = path + x

        return x
