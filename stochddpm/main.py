import copy

import torch
import argparse
import torchvision
import os

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.unet import UNet
from ncsn_cont import NCSN
from models.sampler import Sampler

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=50, help='number of epochs of training')
    parser.add_argument("--in_channels", type=int, default=1, help='number of channels in image')
    parser.add_argument("--b_size", type=int, default=128, help='size of the mini batch')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--img_size', type=float, default=32, help='size of input image')
    parser.add_argument('--data_path', type=str, default='../data', help='path to downloaded dataset')
    parser.add_argument('--conv_dim', nargs='+', type=int, help='channel size', default=32)
    parser.add_argument('--ch_mults', type=list, help='scale factor for conv dim', default=[1, 2, 2, 2])
    parser.add_argument('--n_noise', nargs='+', type=int, help='number of different level of noise', default=1000)
    parser.add_argument('--n_valid', type=int, default=512, help='number of samples to validate')
    parser.add_argument('--parallel', type=bool, default=False, help='use DataParallel')
    parser.add_argument('--from_check', type=int, default=0, help='number of checkpoint, if false - 0')
    parser.add_argument('--n_eval', type=int, default=10000, help='step to evaluate')
    opt = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    count = torch.cuda.device_count()
    print(f'Number of GPU devices is {count}')

    # Params
    conv_dim = opt.conv_dim
    in_channels = opt.in_channels
    img_size = opt.img_size
    b_size = opt.b_size
    n_steps = opt.n_steps
    parallel = opt.parallel
    ch_mults = opt.ch_mults
    data_path = opt.data_path
    n_noise = opt.n_noise
    from_check = opt.from_check
    n_eval = opt.n_eval
    n_valid = opt.n_valid
    sigma_max = 50

    if count > 1:
        parallel = True
        # b_size = b_size * count

    # ------------
    # Data preparation
    # ------------
    trainset = torchvision.datasets.MNIST(root=data_path, train=True,
                                          download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Resize([img_size, img_size]),
                                              torchvision.transforms.Normalize(
                                                  (0.5,), (0.5,))
                                          ])
                                          )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size,
                                              shuffle=True, num_workers=8)

    # --------
    # Model preparation
    # -------
    unet = UNet(in_channels=in_channels,
                n_channels=conv_dim,
                ch_mults=ch_mults).to(device)
    sampler = Sampler(score_nn=unet,
                      device=device)

    if from_check:
        checkpoint = torch.load(f'{data_path}/models_check/ddpm_iter{from_check}.pkl', map_location='cpu')
        if parallel:
            checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
        unet.load_state_dict(checkpoint)

    if parallel:
        unet = torch.nn.DataParallel(unet)

    ncsn = NCSN(score_nn=unet,
                copy_score_nn=copy.deepcopy(unet),
                sampler=sampler,
                data_path=data_path,
                sigma_max=sigma_max,
                device=device)

    # --------
    # Training part
    # -------

    out = ncsn.fit(trainloader=trainloader,
                   n_steps=n_steps)

    # --------
    # Validation part
    # -------

    dir = f'{data_path}/sampling/ncsn2'
    if not os.path.exists(dir):
        os.mkdir(dir)

    path = f'{data_path}/sampling/ncsn2'

    i = 0
    for _ in range(int(n_valid / b_size)):
        size = (b_size, in_channels, img_size, img_size)
        z = torch.randn(size).to(device) * sigma_max
        objects = ncsn.p_sample(z)
        with torch.no_grad():
            img = objects.to('cpu')
            save_image(img.float(), "{}/{}.png".format(path, i))
            i += 1
