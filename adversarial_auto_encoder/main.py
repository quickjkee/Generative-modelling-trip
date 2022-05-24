import torch
import matplotlib.pyplot as plt
import argparse

from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader

from aae import AAE
from models import Discriminator, Decoder, Encoder

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10, help='number of epochs of training')
    parser.add_argument("--b_size", type=int, default=16, help='size of the mini batch')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--img_size', type=float, default=128, help='size of input image')
    parser.add_argument('--data_path', type=str, default='data', help='path of downloaded data')
    parser.add_argument('--h_dim', type=int, default=32, help='dimension of latent code')
    parser.add_argument('--n_layers', type=int, default=2, help='number of discriminator layers')
    parser.add_argument('-conv_dims', '--conv_dims', nargs='+', type=int,
                        help='list of channels for encoder/decoder creation',
                        default=[16, 32, 64, 64, 256, 512])
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

    # ------------
    # Data preparation
    # ------------

    data_train = datasets.MNIST(
        root='data',
        train=True,
        transform=transforms.Compose([ToTensor(),
                                      transforms.Resize((img_size, img_size)),
                                      transforms.Normalize((0.5,), (0.5,))]),
        download=True,
    )

    data_test = datasets.MNIST(
        root='data',
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

    vae = AAE(Encoder=Encoder,
              Decoder=Decoder,
              Discriminator=Discriminator,
              hidden_dim=hidden_dim,
              conv_dims=conv_dims,
              n_layers=n_layers,
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

    # Reconstruction
    for sample in trainloaders:
        out = vae(sample[0].to(device))
        with torch.no_grad():
            for o in out:
                img = torch.reshape(o.to('cpu'), (img_size, img_size))
                plt.imshow(img)
                plt.show()
        break

    # Sampling from noise
    noise = torch.randn(150, hidden_dim).to(device)
    objects = vae.decoder.sample(noise)
    with torch.no_grad():
        for obj in objects:
            img = torch.reshape(obj.to('cpu'), (img_size, img_size))
            plt.imshow(img)
            plt.show()
