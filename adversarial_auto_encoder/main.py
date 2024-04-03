import torch
import matplotlib.pyplot as plt
import argparse
import os
import torchvision

from torch.utils.data import DataLoader

from aae import AAE
from models import Discriminator, Decoder, Encoder

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10, help='number of epochs of training')
    parser.add_argument("--b_size", type=int, default=128, help='size of the mini batch')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--img_size', type=float, default=32, help='size of input image')
    parser.add_argument('--data_path', type=str, default='../data', help='path of downloaded data')
    parser.add_argument('--h_dim', type=int, default=32, help='dimension of latent code')
    parser.add_argument('--n_layers', type=int, default=2, help='number of discriminator layers')
    parser.add_argument('-conv_dims', '--conv_dims', nargs='+', type=int,
                        help='list of channels for encoder/decoder creation',
                        default=[128, 256, 256, 512])
    parser.add_argument('--n_valid', type=int, default=10000, help='number of samples from noise')
    opt = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Params
    n_layers = opt.n_layers
    conv_dims = opt.conv_dims
    img_size = opt.img_size
    hidden_dim = opt.h_dim
    b_size = opt.b_size
    n_epochs = opt.n_epochs
    data_path = opt.data_path
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

    aae = AAE(Encoder=Encoder,
              Decoder=Decoder,
              Discriminator=Discriminator,
              hidden_dim=hidden_dim,
              conv_dims=conv_dims,
              n_layers=n_layers,
              device=device)

    # --------
    # Training part
    # -------

    out = aae.fit(trainloader=trainloader,
                  testloader=testloader,
                  epochs=n_epochs)

    # --------
    # Validation part
    # -------

    dir = f'{data_path}/sampling/aae'
    if not os.path.exists(dir):
        os.mkdir(dir)

    path = f'{data_path}/sampling/aae'

    i = 0
    for _ in range(int(n_valid / b_size)):
        noise = torch.randn(b_size, hidden_dim).to(device)
        objects = aae.decoder.sample(noise)
        with torch.no_grad():
            for obj in objects:
                img = torch.reshape(obj.to('cpu'), (img_size, img_size))
                plt.imsave("{}/{}.png".format(path, i), img, cmap="gray_r")
                i += 1
