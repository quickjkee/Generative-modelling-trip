from gan_model import GAN
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize
from torchvision import transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':
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

    input_size = 128
    output_size = 28 * 28

    gan = GAN(input_size=input_size,
              output_size=output_size,
              epochs=20)

    out = gan.fit(loaders)

    for img in out:
        img = torch.reshape(img, (28, 28))
        plt.imshow(img)
        plt.show()
