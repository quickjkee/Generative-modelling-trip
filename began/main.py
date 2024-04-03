import torch
import argparse
import matplotlib.pyplot as plt
import os
import torchvision

from torch.utils.data import DataLoader

from began import Began
from models import Autoencoder, Generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10, help='number of epochs of training')
    parser.add_argument("--b_size", type=int, default=128, help='size of the mini batch')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--img_size', type=int, default=32, help='size of input image')
    parser.add_argument('--h_dim', type=int, default=32, help='hidden dimension')
    parser.add_argument('--conv_dim', type=int, default=64, help='number of channels in models')
    parser.add_argument('--data_path', type=str, default='../data', help='path of downloaded data')
    parser.add_argument('--in_channels', type=int, default=1, help='number of channels in input img')
    parser.add_argument('--n_valid', type=int, default=10000, help='number of samples from noise')

    opt = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Params
    img_size = opt.img_size
    h_dim = opt.h_dim
    b_size = opt.b_size
    n_epochs = opt.n_epochs
    data_path = opt.data_path
    conv_dim = opt.conv_dim
    in_channels = opt.in_channels
    n_valid = opt.n_valid

    # ------------
    # Data preparation
    # ------------

    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(data_path, train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Resize([img_size, img_size]),
                                       torchvision.transforms.Normalize(
                                           (0.5,), (0.5,))
                                   ])),
        batch_size=b_size, shuffle=True)

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(data_path, train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Resize([img_size, img_size]),
                                       torchvision.transforms.Normalize(
                                           (0.5,), (0.5,))
                                   ])),
        batch_size=b_size, shuffle=True)

    # --------
    # Model preparation
    # -------
    ae = Autoencoder(img_size=img_size,
                     in_channels=in_channels,
                     z_dim=h_dim,
                     n_conv=conv_dim)
    gen = Generator(img_size=img_size,
                    in_channels=in_channels,
                    z_dim=h_dim,
                    n_conv=conv_dim)

    began = Began(ae=ae,
                  gen=gen,
                  z_dim=h_dim,
                  device=device)

    out = began.fit(trainloader,
                    n_epochs)

    # --------
    # Validation part
    # -------

    dir = f'{data_path}/sampling/began'
    if not os.path.exists(dir):
        os.mkdir(dir)

    path = f'{data_path}/sampling/began'

    i = 0
    for _ in range(int(n_valid / b_size)):
        noise = torch.randn(b_size, h_dim).to(device)
        objects = began.sample(noise)
        with torch.no_grad():
            for obj in objects:
                img = torch.reshape(obj.to('cpu'), (img_size, img_size))
                plt.imsave("{}/{}.png".format(path, i), img, cmap="gray_r")
                i += 1
