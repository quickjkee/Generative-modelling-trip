import torch.nn as nn
import torch

class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)

class Energy(nn.Module):
    """
    Approximation of the energy in Boltzmann distribution using CNN
        P = e^(E(X)) / Z - energy based model (Boltzmann distribution)
    Here we approximate E(X)
    """

    def __init__(self, conv_dims, in_channels):
        """
        :param in_channels: (Int), number of channels in input image
        :param conv_dims: (List), number of channels in convolutional layers
        """
        super(Energy, self).__init__()

        self.conv_dims = conv_dims
        self.in_channels = in_channels

        self.model = self._model_init()

    def _model_init(self):
        """
        Initialization of the model
        :return: (nn.Module)
        """
        model = []
        in_channels = self.in_channels

        # Convolutional layers
        for conv in self.conv_dims:
            model.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=conv,
                              kernel_size=3,
                              stride=2,
                              padding=1),
                    Swish(),
                )
            )
            in_channels = conv

        # GlobalMaxPooling and one fully connected layer
        model.append(
            nn.Sequential(
                nn.AdaptiveMaxPool2d(output_size=1),
                nn.Flatten(),
                nn.Linear(self.conv_dims[-1], 128),
                Swish(),
                nn.Linear(128, 1)
            )
        )

        model = nn.Sequential(*model)

        return model

    def forward(self, x):
        """
        Forward propagation of approximation of the energy
        :param x: (Tensor), [b_size x C x W x H], input batch of images
        :return: (Tensor), [b_size x 1], energy values for batch sample
        """
        if len(x.size()) != 4 or x.size()[1] != self.in_channels:
            raise ValueError(
                'ValueError exception thrown, dimension of input have a different form from this [b_size x channels x W x H]')

        output = self.model(x)

        return output
