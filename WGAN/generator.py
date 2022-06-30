import torch.nn as nn


class View(nn.Module):
    """
    Class of view method
    """

    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def __call__(self, x):
        return x.view(*self.shape)


class Generator(nn.Module):
    """
    The realizaton of the generator in wgan model.
    This model generates image from noise
    """

    def __init__(self, h_dim, conv_dims, n_channels):
        """
        :param conv_dims: (List), number of channels for convolutional layers
        :param h_dim: (Integer), dimension of the latent space
        :param n_channels (Integer), number of image channels
        """
        super(Generator, self).__init__()

        conv_dims.reverse()

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
        Initializating of the generator
        :return: (Module), pytorch model
        """
        model = []
        in_channels = self.h_dim

        for i in range(len(self.conv_dims)):
            model.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_channels,
                                       out_channels=self.conv_dims[i],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.BatchNorm2d(num_features=self.conv_dims[i]),
                    nn.ReLU(True)
                )
            )
            in_channels = self.conv_dims[i]

        model.append(nn.ConvTranspose2d(in_channels=self.conv_dims[-1],
                                        out_channels=self.n_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1))
        model.append(nn.Tanh())

        model = nn.Sequential(*model)

        return model

    def forward(self, x):
        """
        Forward pass of generator
        :param x: (Tensor), [b_size x h_dim]
        :return: (Tensor), [b_size x C x W x H]
        """
        if len(x.size()) != 4 or x.size()[1] != self.h_dim:
            raise ValueError(
                'ValueError exception thrown, dimension of input have a different form from this [b_size x h_dim x 1 x 1]')

        samples = self.model(x)
        return samples
