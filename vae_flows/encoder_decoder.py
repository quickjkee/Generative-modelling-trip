import torch
from torch import nn


class Encoder(nn.Module):
    #####
    # Encoding input signal to sample of hidden conditional distribution q(z|x)
    #####
    def __init__(self, hidden_dim, conv_dims, device):
        """
        Constructing encoder
        :param hidden_dim: (Int) Dimension of hidden space
        :param conv_dims: (Int) List of channels dimensional through conv layers
        :param device: Current working device
        """
        super(Encoder, self).__init__()
        self.device = device

        self.hidden_dim = hidden_dim
        self.conv_dims = conv_dims

        model = []
        in_channel = 1  # Working with one dimensional input channel

        for channels in self.conv_dims:
            model.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channel,
                              out_channels=channels,
                              kernel_size=3,
                              stride=2),
                    nn.BatchNorm2d(channels),
                    nn.ReLU())
            )
            in_channel = channels

        model.append(nn.AdaptiveMaxPool2d(output_size=1))
        model.append(nn.Flatten())

        self.model = nn.Sequential(*model)

        self.mu = nn.Linear(in_features=conv_dims[-1],
                            out_features=hidden_dim)  # Expectation of Q(Z|X, Phi) distribution
        self.log_sigma = nn.Linear(in_features=conv_dims[-1],
                                   out_features=hidden_dim)  # Log variance of Q(Z|X, Phi) distribution

    def forward(self, x):
        """
        :param x: (Tensor) [B x C x W x H]
        :return: Tuple(Tensor, Tensor) Tuple of expectation and logarithm of variance (dispersion)
        """
        conv_out = self.model(x)
        mu = self.mu(conv_out)
        log_sigma = self.log_sigma(conv_out)

        return mu, log_sigma

    def sample(self, mu, log_sigma):
        """
        Return sample from latent posterior distribution using reparameterization trick
        :param mu: (Tensor) [B x hidden_dim] Expectation of latent distribution
        :param log_sigma: (Tensor) [B x hidden_dim] Log of variance of latent distribution
        :return: (Tensor) [B x hidden_dim]
        """
        # Latent representation of input object
        assert self.mu is not None, 'Latent distribution is not prepared'

        # Samples from standard normal distribution
        eps = torch.randn(mu.size(dim=0), self.hidden_dim).to(self.device)

        latent_samples = torch.exp(0.5 * log_sigma) * eps + mu

        return latent_samples


class Decoder(nn.Module):
    ######
    # Decoding sample from conditional distribution q(z|x) to input like signal
    #####
    def __init__(self, hidden_dim, conv_dims, device):
        """
        Constructing decoder
        :param hidden_dim: (Int) Dimension of hidden space
        :param conv_dims: (Int) List of channels dimensional through conv layers
        :param device: Current working device
        """
        super(Decoder, self).__init__()

        self.device = device

        self.hidden_dim = hidden_dim

        conv_dims.reverse()  # Reversing dims to create decoder
        self.conv_dims = conv_dims  # It is possible to decrease number of parameters for prevent overfitting

        self.input_layer = nn.Linear(self.hidden_dim, self.conv_dims[0] * 16)

        """
        Each model`s layer will upscale input image size by two
        We are starting with 4x4 image`s size. Thus, we need
        five conv2transpose layers
        """
        model = []
        for i in range(len(self.conv_dims) - 1):
            model.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=self.conv_dims[i],
                                       out_channels=self.conv_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1
                                       ),
                    nn.BatchNorm2d(self.conv_dims[i + 1]),
                    nn.ReLU()
                )
            )
        self.model = nn.Sequential(*model)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.conv_dims[-1],
                               out_channels=self.conv_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1
                               ),
            nn.BatchNorm2d(self.conv_dims[i + 1]),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.conv_dims[-1],
                      out_channels=1,
                      kernel_size=3,
                      padding=1
                      ),
            nn.Tanh()
        )

    def forward(self, z):
        """
        Decoding sample from latent distribution
        :param z: (Tensor) [B x hidden_dim]
        :return: (Tensor) [B x C x W x H]
        """
        input = self.input_layer(z).view(-1, self.conv_dims[0], 4, 4)
        out_decoder = self.model(input)
        out_final = self.final_layer(out_decoder)

        return out_final
