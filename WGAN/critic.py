import torch.nn as nn


class Critic(nn.Module):
    """
    The critic realization in the wgan
    """

    def __init__(self, h_dim, conv_dims, n_channels):
        super(Critic, self).__init__()
        """
        :param conv_dims: (List), number of channels for convolutional layers
        :param h_dim: (Integer), dimension of the latent space
        :param n_channels (Integer), number of image channels
        """
        super(Critic, self).__init__()

        self.h_dim = h_dim
        self.conv_dims = conv_dims
        self.n_channels = n_channels

        self.model = self._model_create()
        self._weights_init()

    def _weights_init(self):
        """
        Model parameters initializing
        :return: None
        """
        classname = self.model.__class__.__name__

        if classname.find('Conv') != -1:
            nn.init.normal_(self.model.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(self.model.weight.data, 1.0, 0.02)
            nn.init.constant_(self.model.bias.data, 0)

    def _model_create(self):
        """
        Initialization of the critic
        :return: (Model), pytorch model
        """
        model = []

        in_channels = self.n_channels
        for dim in self.conv_dims:
            model.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=dim,
                              kernel_size=4,
                              stride=2,
                              padding=1),
                    nn.BatchNorm2d(num_features=dim),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_channels = dim

        model.append(
            nn.Sequential(
                nn.Conv2d(in_channels=self.conv_dims[-1],
                          out_channels=1,
                          kernel_size=4,
                          stride=2,
                          padding=1),
            )
        )
        model = nn.Sequential(*model)

        return model

    def forward(self, x):
        """
        Forward pass
        :param x: (Tensor), [b_size x C x W x H]
        :return: (Tensor), [b_size x 1]
        """
        if len(x.size()) != 4:
            raise ValueError(
                'ValueError exception thrown, dimension of input have a different form from this [b_size x C x W x H]')
        out = self.model(x)

        return out
