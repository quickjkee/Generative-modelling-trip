import torch
import argparse
import matplotlib.pyplot as plt
import torchvision
import os

from torch.utils.data import DataLoader

from annealed_langevin import sample_anneal_langevin
from models import ScoreNetwork
from ncsn import NCSN

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10, help='number of epochs of training')
    parser.add_argument("--in_channels", type=int, default=1, help='number of channels in image')
    parser.add_argument("--b_size", type=int, default=64, help='size of the mini batch')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--img_size', type=float, default=32, help='size of input image')
    parser.add_argument('--data_path', type=str, default='../data', help='path of downloaded data')
    parser.add_argument('--conv_dims', nargs='+', type=int, help='channel size', default=128)
    parser.add_argument('--n_valid', type=int, default=10000, help='number of samples from noise')
    opt = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Params
    conv_dims = opt.conv_dims
    in_channels = opt.in_channels
    img_size = opt.img_size
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

    score_nn = ScoreNetwork(channels=conv_dims,
                            in_channels=in_channels,
                            out_channels=in_channels)
    sampler = sample_anneal_langevin
    ncsn = NCSN(score_nn=score_nn,
                sampler=sampler,
                device=device,
                sigmas=torch.linspace(1, 0.01, steps=10))

    # --------
    # Training part
    # -------

    out = ncsn.fit(trainloader=trainloader,
                   n_epochs=n_epochs)

    # --------
    # Validation part
    # -------

    dir = f'{data_path}/sampling/ncsn'
    if not os.path.exists(dir):
        os.mkdir(dir)

    path = f'{data_path}/sampling/ncsn'

    i = 0
    for _ in range(int(n_valid / b_size)):
        size = (b_size, in_channels, img_size, img_size)
        objects = ncsn.sample(size)[-1]
        with torch.no_grad():
            for obj in objects:
                img = torch.reshape(obj.to('cpu'), (img_size, img_size))
                plt.imsave("{}/{}.png".format(path, i), img, cmap="gray_r")
                i += 1
