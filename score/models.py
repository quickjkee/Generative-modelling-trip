import torch.nn.functional as F
import torch.nn as nn


class CondBatchNorm2D(nn.Module):
    """
    Realization of the conditional batch normalization
    """

    def __init__(self, num_features, num_classes):
        """
        :param num_features: (Int), number of channels in image
        :param num_classes: (Int), number of different noise level
        """
        super(CondBatchNorm2D, self).__init__()

        self.num_features = num_features

        self.embed = nn.Embedding(num_embeddings=num_classes,
                                  embedding_dim=num_features * 2)
        self.norm = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x, y):
        out = self.norm(x)
        gamma, beta = self.embed(y).chunk(2, dim=1)

        out = out * gamma.view(-1, self.num_features, 1, 1) + beta.view(-1, self.num_features, 1, 1)

        return out


class ScoreNetwork(nn.Module):
    """
    Realization of the ScoreNetwork.
    """

    def __init__(self, channels, num_noise=10, in_channels=1, out_channels=1):
        """
        Initialization of the Score network
        :param num_noise: (Int), number of noise levels
        :param channels: (Int), channel size
        :param in_channels: (Int), input channels of the image
        :param out_channels: (Int), output channels of the image
        """
        super(ScoreNetwork, self).__init__()

        self.channels = channels
        self.num_noise = num_noise

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn0 = CondBatchNorm2D(num_features=in_channels, num_classes=self.num_noise)

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1,
                               stride=1)
        self.bn1 = CondBatchNorm2D(num_features=channels, num_classes=self.num_noise)

        self.conv2 = nn.Conv2d(in_channels=channels,
                               out_channels=channels * 2,
                               kernel_size=3,
                               padding=1,
                               stride=1)
        self.bn2 = CondBatchNorm2D(num_features=channels * 2, num_classes=self.num_noise)

        self.conv3 = nn.Conv2d(in_channels=channels * 2,
                               out_channels=channels * 3,
                               kernel_size=3,
                               padding=1,
                               stride=1)
        self.bn3 = CondBatchNorm2D(num_features=channels * 3, num_classes=self.num_noise)

        self.conv4 = nn.Conv2d(in_channels=channels * 3,
                               out_channels=out_channels,
                               kernel_size=3,
                               padding=1,
                               stride=1)

    def forward(self, x, y):
        """
        Forward propagation of the Score Network
        :param x: (Tensor), [b_size x C_in x W x H]
        :param y: (Tensor), [b_size]
        :return: (Tensor), [b_size x C_out x W x H]
        """
        out = self.bn0(x, y)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn1(out, y)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, y)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out, y)
        out = F.relu(out)
        out = self.conv4(out)

        return out
