import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize
from torchvision import transforms
from torch.utils.data import DataLoader
from vae_model import VAE

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conv_dims = [16, 32, 64, 128]
    IMG_SIZE = 32

    data_train = datasets.MNIST(
        root='data',
        train=True,
        transform=transforms.Compose([ToTensor(),
                                      transforms.Resize((IMG_SIZE, IMG_SIZE))]),
        download=True,
    )

    data_test = datasets.MNIST(
        root='data',
        train=False,
        transform=transforms.Compose([ToTensor(),
                                      transforms.Resize((IMG_SIZE, IMG_SIZE))]),
        download=True,
    )

    trainloaders = DataLoader(data_train,
                              batch_size=32,
                              shuffle=True,
                              num_workers=1)

    testloaders = DataLoader(data_test,
                             batch_size=32,
                             shuffle=True,
                             num_workers=1)

    hidden_dim = 16
    data_dim = 32 * 32

    vae = VAE(data_dim=data_dim,
              hidden_dim=hidden_dim,
              conv_dims=conv_dims,
              device=device)

    out = vae.fit(trainloader=trainloaders,
                  testloader=testloaders,
                  epochs=20)

    # Reconstruction
    for sample in trainloaders:
        out = vae(sample[0].to(device))
        with torch.no_grad():
            for o in out:
                img = torch.reshape(o.to('cpu'), (IMG_SIZE, IMG_SIZE))
                plt.imshow(img)
                plt.show()
        break

    # Sampling from noise
    noise = torch.randn(150, hidden_dim).to(device)
    objects = vae.decoder.sample(noise)
    with torch.no_grad():
        for obj in objects:
            img = torch.reshape(obj.to('cpu'), (IMG_SIZE, IMG_SIZE))
            plt.imshow(img)
            plt.show()
