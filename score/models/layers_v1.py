import torch.nn as nn
import torch


##############################
#
# Conditional Residual Network
#
##############################

class CondResBlock(nn.Module):
    def __init__(self, n_classes, in_channels, out_channels, dilation=None, resize=False):
        """
        :param n_classes: (Int), number of different levels of noise
        :param in_channels: (Int), number of input channels
        :param out_channels: (Int), number of output channels
        """
        super(CondResBlock, self).__init__()

        self.act = nn.ELU()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if not resize:
            if dilation:
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation)
            else:
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = None

        else:
            self.shortcut = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                CondBatchNorm2d(n_classes=n_classes,
                                n_features=out_channels)
            ])
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, dilation=dilation,
                                   padding=dilation)

        self.norm1 = CondBatchNorm2d(n_classes=n_classes,
                                     n_features=in_channels)
        self.norm2 = CondBatchNorm2d(n_classes=n_classes,
                                     n_features=in_channels)

    def forward(self, x, y):
        """
        :param x: (Tensor), [b_size x C x H x W], input image
        :param y: (Tensor), [b_size x 1], labels of noise
        :return: (Tensor), [b_size x C' x H/2 x W/2], output image
        """
        # (B x C x W x H) -> (B x C x W/2 x H/2)
        if self.shortcut:
            shortcut = self.shortcut[0](x)
            shortcut = self.shortcut[1](shortcut, y)
        else:
            shortcut = x

        # (B x C x W x H) -> (B x C x W x H)
        out = self.act(self.conv1(self.norm1(x), y))

        # (B x C x W x H) -> (B x C x W/2 x H/2)
        out = self.act(self.conv2(self.norm2(out), y))
        out = out + shortcut

        return self.act(out)


class CondBatchNorm2d(nn.Module):
    def __init__(self, n_features, n_classes, bias=True):
        super().__init__()
        self.n_features = n_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(n_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(n_classes, n_features * 3)
            self.embed.weight.data[:, :2 * n_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, 2 * n_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(n_classes, 2 * n_features)
            self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.n_features, 1, 1) * h + beta.view(-1, self.n_features, 1, 1)
        else:
            gamma, alpha = self.embed(y).chunk(2, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.n_features, 1, 1) * h
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
        self.norm1 = nn.InstanceNorm2d(features)
        self.norm2 = nn.InstanceNorm2d(features)

    def forward(self, x):
        # (B x C x W x H) -> (B x C x W x H)
        out = self.norm1(x)
        out = self.conv1(self.act(out))
        out = self.norm2(out)
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
                nn.InstanceNorm2d(self.in_planes[i]),
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
            convs.append(nn.Sequential(nn.InstanceNorm2d(self.features),
                                       nn.Conv2d(self.features,
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
