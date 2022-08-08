import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, img_size, z_dim, in_channels, n_conv):
        """
        Transformation from image to latent space
        :param z_dim: (Int), dimension of latent space
        :param n_conv: (Int), number of filters in conv layers
        :param in_channels: (Int), number of channels in input image
        """
        super(Encoder, self).__init__()

        self.img_size = img_size
        self.z_dim = z_dim

        self.in_channels = in_channels
        self.n_conv = n_conv

        self.model = self._model_create()

    def _model_create(self):
        out_size = int(self.img_size / 4)

        model = nn.Sequential(
            # (B x C_in x W X H) -> (B x C x W x H)
            nn.Conv2d(self.in_channels, self.n_conv, 3, 1, 1),
            nn.ELU(True),

            # (B x C x W X H) -> (B x 2*C x W/2 x H/2)
            nn.Conv2d(self.n_conv, 2 * self.n_conv, 3, 2, 1),
            nn.ELU(True),

            # (B x 2*C x W/2 x H/2) -> (B x 2*C x W/2 x H/2)
            nn.Conv2d(2 * self.n_conv, 2 * self.n_conv, 3, 1, 1),
            nn.ELU(True),

            # (B x 2*C x W/2 x H/2) -> (B x 3*C x W/4 x H/4)
            nn.Conv2d(2 * self.n_conv, 3 * self.n_conv, 3, 2, 1),
            nn.ELU(True),

            # (B x 3*C x W/4 x H/4) -> (B x 3*C x W/4 x H/4)
            nn.Conv2d(3 * self.n_conv, 3 * self.n_conv, 3, 1, 1),
            nn.ELU(True),

            # (B x 2*C x W/2 x H/2) -> (B x 3*C x W/4 x H/4)
            nn.Conv2d(3 * self.n_conv, 3 * self.n_conv, 3, 1, 1),
            nn.ELU(True),

            # (B x 3*C x W/4 x H/4) -> (B x z_dim)
            nn.Flatten(),
            nn.Linear(out_size * out_size * 3 * self.n_conv, self.z_dim)
        )

        return model

    def forward(self, x):
        """
        Convert x to latent space
        :param x: (Tensor), [b_size x C x W x H]
        :return: (Tensor), [b_size x z_dim]
        """
        if x.size(dim=2) != self.img_size:
            raise ValueError("Input size of image does not matches with initialized")
        out = self.model(x)

        return out


class Decoder(nn.Module):
    def __init__(self, z_dim, n_conv, in_channels):
        """
        Transformation from latent space to image
        """
        super(Decoder, self).__init__()

        self.z_dim = z_dim

        self.in_channels = in_channels
        self.n_conv = n_conv

        # (B x z_dim) -> # (B x 8*8*n_conv)
        self.in_layer = nn.Linear(z_dim, 8 * 8 * self.n_conv)
        self.model = self._model_create()

    def _model_create(self):
        model = nn.Sequential(
            # (B x n_conv x 8 x 8) -> # (B x n_conv x 8 x 8)
            nn.Conv2d(self.n_conv, self.n_conv, 3, 1, 1),
            nn.ELU(True),

            # (B x n_conv x 8 x 8) -> # (B x 2*n_conv x 16 x 16)
            nn.Conv2d(self.n_conv, 2 * self.n_conv, 3, 1, 1),
            nn.ELU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            # (B x 2*n_conv x 16 x 16) -> # (B x 2*n_conv x 16 x 16)
            nn.Conv2d(2 * self.n_conv, 2 * self.n_conv, 3, 1, 1),
            nn.ELU(True),

            # (B x 2*n_conv x 16 x 16) -> # (B x 3*n_conv x 32 x 32)
            nn.Conv2d(2 * self.n_conv, 3 * self.n_conv, 3, 1, 1),
            nn.ELU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(3 * self.n_conv, self.in_channels, 3, 1, 1),
        )

        return model

    def forward(self, x):
        """
        Transformation from latent space to image
        :param x: (Tensor), [b_size, z_dim]
        :return: (Tensor), [b_size x C x W x H]
        """
        if x.size(dim=1) != self.z_dim:
            raise ValueError("Input size of latent vector does not matches with initialized")

        input = self.in_layer(x).view(-1, self.n_conv, 8, 8)
        out = self.model(input)

        return out


class Autoencoder(nn.Module):
    def __init__(self, img_size, in_channels, z_dim, n_conv):
        super(Autoencoder, self).__init__()

        self.z_dim = z_dim
        self.img_size = img_size

        self.n_conv = n_conv
        self.in_channels = in_channels

        self.model = self._model_create()

    def _model_create(self):
        model = nn.Sequential(
            Encoder(img_size=self.img_size,
                    z_dim=self.z_dim,
                    in_channels=self.in_channels,
                    n_conv=self.n_conv),
            Decoder(z_dim=self.z_dim,
                    n_conv=self.n_conv,
                    in_channels=self.in_channels),
        )

        return model

    def forward(self, x):
        """
        Autoencoding of input image
        :param x: (Tensor), [b_size x C x W x H]
        :return: (Tensor), [b_size x C x W x H]
        """
        out = self.model(x)

        return out


class Generator(nn.Module):
    def __init__(self, img_size, in_channels, z_dim, n_conv):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.img_size = img_size

        self.n_conv = n_conv
        self.in_channels = in_channels

        self.model = self._model_create()

    def _model_create(self):
        model = nn.Sequential(
            Decoder(z_dim=self.z_dim,
                    n_conv=self.n_conv,
                    in_channels=self.in_channels),
        )

        return model

    def forward(self, x):
        """
        From latent space to image
        :param x: (Tensor), [b_size x z_dim]
        :return: (Tensor), [b_size x C x W x H]
        """
        out = self.model(x)

        return out
