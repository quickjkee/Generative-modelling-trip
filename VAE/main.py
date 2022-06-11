import torch
import argparse
import matplotlib.pyplot as plt
import os

from torchvision import transforms
from torch.utils.data import DataLoader

from vae_model import VAE
from dataset import CellTrapSet, train_test_split

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10, help='number of epochs of training')
    parser.add_argument("--b_size", type=int, default=16, help='size of the mini batch')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--img_size', type=float, default=128, help='size of input image')
    parser.add_argument('--data_path', type=str, default='data', help='path of downloaded data')
    parser.add_argument('--h_dim', type=int, default=32, help='dimension of latent code')
    parser.add_argument('-conv_dims', '--conv_dims', nargs='+', type=int,
                        help='list of channels for encoder/decoder creation',
                        default=[32, 64, 128, 256, 256, 512])
    parser.add_argument('--n_valid', type=int, default=10000, help='number of samples from noise')
    opt = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Params
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

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([img_size, img_size]),
        transforms.Normalize((0.5,), (0.5,))])

    micro_path = data_path + 'microfluidic/'

    names = os.listdir(micro_path + 'images')
    train_names, test_names = train_test_split(names)

    trainset = CellTrapSet(train_names, micro_path, transforms=trans)
    testset = CellTrapSet(test_names, micro_path, transforms=trans)

    trainloaders = DataLoader(trainset, b_size, shuffle=True)
    testloaders = DataLoader(testset, b_size, shuffle=True)

    # --------
    # Model preparation
    # -------

    vae = VAE(hidden_dim=hidden_dim,
              conv_dims=conv_dims,
              device=device)

    # --------
    # Training part
    # -------

    out = vae.fit(trainloader=trainloaders,
                  testloader=testloaders,
                  epochs=n_epochs)

    # --------
    # Validation part
    # -------

    dir = data_path + 'sampling/vae'
    if not os.path.exists(dir):
        os.mkdir(dir)

    path = data_path + 'sampling/vae'

    i = 0
    for _ in range(int(n_valid / b_size)):
        noise = torch.randn(b_size, hidden_dim).to(device)
        objects = vae.decoder.sample(noise)
        with torch.no_grad():
            for obj in objects:
                img = torch.reshape(obj.to('cpu'), (img_size, img_size))
                plt.imsave("{}/{}.png".format(path, i), img, cmap="gray_r")
                i += 1
