import torch
import argparse
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize
from torchvision import transforms
from torch.utils.data import DataLoader

from gan_model import GAN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10, help='number of epochs of training')
    parser.add_argument("--b_size", type=int, default=16, help='size of the mini batch')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--img_size', type=int, default=28, help='size of input image')
    parser.add_argument('--h_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--data_path', type=str, default='data', help='path of downloaded data')

    opt = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Params
    img_size = opt.img_size
    h_dim = opt.h_dim
    b_size = opt.b_size
    n_epochs = opt.n_epochs
    data_path = opt.data_path

    # ------------
    # Data preparation
    # ------------

    data = datasets.MNIST(
        root=data_path,
        train=True,
        transform=transforms.Compose([ToTensor(), Normalize((0.5,), (0.5,))]),
        download=True,
    )

    loaders = DataLoader(data,
                         batch_size=b_size,
                         shuffle=True,
                         num_workers=1)

    # --------
    # Model preparation
    # -------

    gan = GAN(input_size=h_dim,
              output_size=int(img_size * img_size),
              epochs=n_epochs,
              device=device)

    out = gan.fit(loaders)

    for img in out:
        img = torch.reshape(img, (28, 28))
        plt.imshow(img)
        plt.show()
