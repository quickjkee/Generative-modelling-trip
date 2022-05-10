import torch
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize
from torchvision import transforms
from torch.utils.data import DataLoader

from planar_flow import PlanarFlow
from vae_flows import VaeFlow
from encoder_decoder import Encoder, Decoder

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = 128

    data_train = datasets.MNIST(
        root='data',
        train=True,
        transform=transforms.Compose([ToTensor(),
                                      transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                      transforms.Normalize((0.5,), (0.5,))]),
        download=True,
    )

    data_test = datasets.MNIST(
        root='data',
        train=False,
        transform=transforms.Compose([ToTensor(),
                                      transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                      transforms.Normalize((0.5,), (0.5,))]),
        download=True,
    )

    trainloaders = DataLoader(data_train,
                              batch_size=128,
                              shuffle=True,
                              num_workers=1)

    testloaders = DataLoader(data_test,
                             batch_size=128,
                             shuffle=True,
                             num_workers=1)

    # Flow
    hidden_dim = 32
    flow_size = 64
    activation = torch.nn.Tanh()
    der_activation = 1  # Пока так, потом поменять!

    flow = PlanarFlow(len_f=flow_size,
                      h_dim=hidden_dim,
                      activation=activation,
                      der_activation=der_activation)

    # Encoder/Decoder
    conv_dims = [16, 32, 64, 128, 256]

    encoder = Encoder(hidden_dim=hidden_dim,
                      conv_dims=conv_dims,
                      device=device)

    decoder = Decoder(hidden_dim=hidden_dim,
                      conv_dims=conv_dims,
                      device=device)

    # VaeFlow
    vae_flow = VaeFlow(encoder=encoder,
                       decoder=decoder,
                       flow=flow,
                       hidden_dim=hidden_dim,
                       device=device)

    vae_flow.fit(trainloader=trainloaders,
                 testloader=testloaders,
                 epochs=10)
