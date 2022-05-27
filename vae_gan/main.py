import torch
import argparse
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader

from vae_gan import VAE_GAN
from models import Discriminator, Decoder, Encoder

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10, help='number of epochs of training')
    parser.add_argument("--b_size", type=int, default=16, help='size of the mini batch')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--img_size', type=float, default=128, help='size of input image')
    parser.add_argument('--data_path', type=str, default='data', help='path of downloaded data')
    parser.add_argument('--h_dim', type=int, default=16, help='dimension of latent code')
    parser.add_argument('-conv_dims', '--conv_dims', nargs='+', type=int,
                        help='list of channels for encoder/decoder creation',
                        default=[32, 32, 64, 128, 256])
    parser.add_argument('--n_valid', type=int, default=150, help='number of samples from noise')
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

    data_train = datasets.MNIST(
        root=data_path,
        train=True,
        transform=transforms.Compose([ToTensor(),
                                      transforms.Resize((img_size, img_size)),
                                      transforms.Normalize((0.5,), (0.5,))]),
        download=True,
    )

    data_test = datasets.MNIST(
        root=data_path,
        train=False,
        transform=transforms.Compose([ToTensor(),
                                      transforms.Resize((img_size, img_size)),
                                      transforms.Normalize((0.5,), (0.5,))]),
        download=True,
    )

    trainloaders = DataLoader(data_train,
                              batch_size=b_size,
                              shuffle=True,
                              num_workers=1)

    testloaders = DataLoader(data_test,
                             batch_size=b_size,
                             shuffle=True,
                             num_workers=1)

    # --------
    # Model preparation
    # -------

    vae = VAE_GAN(Encoder=Encoder,
                  Decoder=Decoder,
                  Discriminator=Discriminator,
                  hidden_dim=hidden_dim,
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

    path = 'data/Sampling'

    # Sampling from noise
    noise = torch.randn(n_valid, hidden_dim).to(device)
    objects = vae.decoder.sample(noise)
    with torch.no_grad():
        for i, obj in enumerate(objects):
            img = torch.reshape(obj.to('cpu'), (img_size, img_size))
            plt.imsave("{}/{}.png".format(path, i), img, cmap="gray_r")
