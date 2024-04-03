import torch
import argparse
import torchvision
import os

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from annealed_langevin import sample_anneal_langevin
from models.refinenet_v2 import RefineNet
from ncsn import NCSN

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=1000, help='number of epochs of training')
    parser.add_argument("--in_channels", type=int, default=3, help='number of channels in image')
    parser.add_argument("--b_size", type=int, default=128, help='size of the mini batch')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--img_size', type=float, default=32, help='size of input image')
    parser.add_argument('--data_path', type=str, default='../data', help='path of downloaded data')
    parser.add_argument('--conv_dims', nargs='+', type=int, help='channel size', default=128)
    parser.add_argument('--n_noise', nargs='+', type=int, help='number of different level of noise', default=10)
    parser.add_argument('--n_valid', type=int, default=128, help='number of samples from noise')
    opt = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Params
    conv_dims = opt.conv_dims
    in_channels = opt.in_channels
    img_size = opt.img_size
    b_size = opt.b_size
    n_epochs = opt.n_epochs
    data_path = opt.data_path
    n_noise = opt.n_noise
    n_valid = opt.n_valid

    # ------------
    # Data preparation
    # ------------
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Resize([img_size, img_size]),
                                                torchvision.transforms.Normalize(
                                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ])
                                            )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size,
                                              shuffle=True, num_workers=2)

    # --------
    # Model preparation
    # -------

    score_nn = RefineNet(in_channels=in_channels,
                         channels=conv_dims)
    sampler = sample_anneal_langevin
    ncsn = NCSN(score_nn=score_nn,
                sampler=sampler,
                device=device,
                data_path=data_path,
                sigmas=torch.exp(
                    torch.linspace(torch.log(torch.tensor(50).float()), torch.log(torch.tensor(0.01).float()),
                                   steps=232)))

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
        objects = (ncsn.sample(size)[-1] + 1.0) / 2
        with torch.no_grad():
            img = objects.to('cpu')
            save_image(img, "{}/{}.png".format(path, i))
            i += 1
