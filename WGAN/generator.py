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

    def _model_create(self):
        """
        Initializating of the generator
        :return: (Module), pytorch model
        """
        model = []
        in_channels = 1

        model.append(nn.Linear(in_features=self.h_dim,
                               out_features=4))
        model.append(View(shape=(-1, 1, 2, 2)))

        for i in range(len(self.conv_dims)):
            model.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=self.conv_dims[i],
                              kernel_size=3,
                              padding='same'),
                    nn.BatchNorm2d(num_features=self.conv_dims[i]),
                    nn.ReLU()
                )
            )
            in_channels = self.conv_dims[i]

        model.append(nn.Conv2d(in_channels=self.conv_dims[-1],
                               out_channels=self.n_channels,
                               kernel_size=3,
                               padding='same'))
        model.append(nn.Tanh())

        model = nn.Sequential(*model)

        return model

    def forward(self, x):
        """
        Forward pass of generator
        :param x: (Tensor), [b_size x h_dim]
        :return: (Tensor), [b_size x C x W x H]
        """
        if len(x.size()) != 2 or x.size()[-1] != self.h_dim:
            raise ValueError(
                'ValueError exception thrown, dimension of input have a different form from this [b_size x h_dim]')

        samples = self.model(x)
        return samples
