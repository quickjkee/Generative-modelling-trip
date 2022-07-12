import torch
import torchvision
import argparse
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader

from ebm import EBM
from energy import Energy
from langevin_sampler import Sampler

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10, help='number of epochs of training')
    parser.add_argument("--n_steps", type=int, default=100, help='number of steps in Leapfrog method')
    parser.add_argument("--step_size", type=int, default=0.1, help='step size in Leapfrog method')
    parser.add_argument("--n_channels", type=int, default=1, help='number of channels in image')
    parser.add_argument("--b_size", type=int, default=16, help='size of the mini batch')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--img_size', type=float, default=32, help='size of input image')
    parser.add_argument('--data_path', type=str, default='../data', help='path of downloaded data')
    parser.add_argument('-conv_dims', '--conv_dims', nargs='+', type=int,
                        help='list of channels for encoder/decoder creation',
                        default=[8, 16, 32, 32, 64, 128])
    parser.add_argument('--n_valid', type=int, default=10000, help='number of samples for validation after training')
    opt = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Params
    conv_dims = opt.conv_dims
    n_channels = opt.n_channels
    img_size = opt.img_size
    b_size = opt.b_size
    n_epochs = opt.n_epochs
    n_steps = opt.n_steps
    step_size = opt.step_size
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

    sampler = Sampler(n_samples=b_size,
                      n_steps=n_steps,
                      step_size=step_size,
                      img_shape=(n_channels, img_size, img_size),
                      device=device)

    energy = Energy(conv_dims=conv_dims,
                    in_channels=n_channels)

    ebm = EBM(sampler=sampler,
              energy=energy,
              device=device)

    # --------
    # Training part
    # -------

    out = ebm.fit(trainloader=trainloader,
                  testloader=testloader,
                  n_epochs=n_epochs)

    # --------
    # Validation part
    # -------

    dir = data_path + 'sampling/ebm'
    if not os.path.exists(dir):
        os.mkdir(dir)

    path = data_path + 'sampling/ebm'

    i = 0
    for _ in range(int(n_valid / b_size)):
        objects = ebm.sample()
        with torch.no_grad():
            for obj in objects:
                img = torch.reshape(obj.to('cpu'), (img_size, img_size))
                plt.imsave("{}/{}.png".format(path, i), img, cmap="gray_r")
                i += 1
