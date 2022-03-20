from vae_model import VAE
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize
from torchvision import transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = datasets.MNIST(
        root='data',
        train=True,
        transform=transforms.Compose([ToTensor(), Normalize((0.5,), (0.5,))]),
        download=True,
    )

    loaders = DataLoader(data,
                         batch_size=100,
                         shuffle=True,
                         num_workers=1)

    hidden_dim = 64
    data_dim = 28 * 28

    vae = VAE(data_dim=data_dim,
              hidden_dim=hidden_dim,
              device=device)

    out = vae.fit(trainloader=loaders,
                  testloader=loaders,
                  epochs=50)

    # Sampling
    noise = torch.randn(5, hidden_dim)
    objects = vae.decoder.sample(noise)
    with torch.no_grad():
        for obj in objects:
            img = torch.reshape(obj, (28, 28))
            plt.imshow(img)
            plt.show()