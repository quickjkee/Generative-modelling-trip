import torch
import argparse
import matplotlib.pyplot as plt
import os
import torchvision

from torch.utils.data import DataLoader

from gan_model import GAN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=1, help='number of epochs of training')
    parser.add_argument("--b_size", type=int, default=128, help='size of the mini batch')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--img_size', type=int, default=32, help='size of input image')
    parser.add_argument('--h_dim', type=int, default=16, help='hidden dimension')
    parser.add_argument('--data_path', type=str, default='../data', help='path of downloaded data')
    parser.add_argument('--n_valid', type=int, default=10000, help='number of samples from noise')

    opt = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Params
    img_size = opt.img_size
    h_dim = opt.h_dim
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

    gan = GAN(input_size=h_dim,
              output_size=int(img_size * img_size),
              epochs=n_epochs,
              device=device)

    out = gan.fit(trainloader)

    # --------
    # Validation part
    # -------

    dir = f'{data_path}/sampling/gan'
    if not os.path.exists(dir):
        os.mkdir(dir)

    path = f'{data_path}/sampling/gan'

    i = 0
    for _ in range(int(n_valid / b_size)):
        noise = torch.randn(b_size, h_dim).to(device)
        objects = gan(noise)
        with torch.no_grad():
            for obj in objects:
                img = torch.reshape(obj.to('cpu'), (img_size, img_size))
                plt.imsave("{}/{}.png".format(path, i), img, cmap="gray_r")
                i += 1
