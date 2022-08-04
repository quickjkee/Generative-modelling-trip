import torch
from torch import nn


class Generator(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.gen = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(1024, output_size),
            nn.Tanh(),
        )

    def forward(self, x):
        assert x.size(dim=1) == self.input_size, 'Dimensional of input noise should equal initialized'

        out = self.gen(x)

        return out


class Discriminator(torch.nn.Module):

    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.disc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(128, 512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.disc(x)

        return out


class GAN(torch.nn.Module):

    def __init__(self, input_size, output_size, epochs, device):
        super(GAN, self).__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.generator = Generator(input_size=input_size,
                                   output_size=output_size).to(self.device)
        self.discriminator = Discriminator(input_size=output_size).to(self.device)
        self.bce = torch.nn.BCELoss()

        self.g_optimizer = torch.optim.Adam(params=self.generator.parameters(), lr=3e-4)
        self.d_optimizer = torch.optim.Adam(params=self.discriminator.parameters(), lr=3e-4)
        self.epochs = epochs

    def __call__(self, X):
        # input noise
        out = self.generator(X)

        return out

    def discriminator_loss(self, pred_true, pred_fake):
        targets_true = torch.full(size=(pred_true.size(dim=0), 1), fill_value=1, dtype=torch.float32).to(self.device)
        targets_fake = torch.full(size=(pred_true.size(dim=0), 1), fill_value=0, dtype=torch.float32).to(self.device)

        bce_true = self.bce(pred_true, targets_true)
        bce_fake = self.bce(pred_fake, targets_fake)

        bce = (bce_fake + bce_true) / 2

        return bce

    def generator_loss(self, pred_fake):
        targets_fake = torch.full(size=(pred_fake.size(dim=0), 1), fill_value=1, dtype=torch.float32).to(self.device)
        bce_generator = self.bce(pred_fake, targets_fake)

        return bce_generator

    def fit(self, dataloader):
        G_losses = []
        D_losses = []
        img_fake = []

        for epoch in range(self.epochs):
            for i, (data, labels) in enumerate(dataloader):
                #########
                # Discriminator train stage

                # minimize -1 * {log(D(x)) + log(1 - D(G(z)))}

                # it equivalent to minimize sum of two binary cross entropy
                # first : - {y_true * log(y = 1 | x) + (1 - y_true) * (1 - log(y = 1 | x))} = - { 1 * log(y = 1 | x) }
                # second: - {y_fake * log(y = 1 | x) + (1 - y_fake) * (1 - log(y = 1 | x))} =
                # = - { (1 - y_fake) * (1 - log(y = 1 | x)) }
                # y_true = 1, y_fake = 0
                #########

                self.discriminator.zero_grad()

                b_size = data.size(0)
                Z_samples = torch.randn(b_size, self.input_size).to(self.device)
                fake_data = self.generator(Z_samples)

                real_data = data.to(self.device).view(-1, self.output_size)
                pred_true = self.discriminator(real_data)
                pred_fake = self.discriminator(fake_data)

                loss_D = self.discriminator_loss(pred_true, pred_fake)
                loss_D.backward()

                self.d_optimizer.step()

                #########
                # Generator train stage

                # minimize log(1 - D(G(z)) ( minimize(-log(D(G(z)) )

                # it equivalent to maximize binary cross entropy
                # - {y_fake * log(y = 1 | x) + (1 - y_fake) * (1 - log(y = 1 | x))} =
                # = - { (1 - y_fake) * (1 - log(y = 1 | x)) }
                # y_fake = 0
                #########

                self.generator.zero_grad()

                Z_samples = torch.randn(b_size, self.input_size).to(self.device)
                fake_data = self.generator(Z_samples)

                pred_fake = self.discriminator(fake_data)

                loss_G = self.generator_loss(pred_fake)
                loss_G.backward()

                self.g_optimizer.step()

                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                          % (epoch, self.epochs, i, len(dataloader),
                             loss_D.item(), loss_G.item()))

                ###################################

                G_losses.append(loss_G.item())
                D_losses.append(loss_D.item())

                ##################################

            if epoch % 2 == 0:
                with torch.no_grad():
                    size = 3
                    noise = torch.randn(size, self.input_size).to(self.device)
                    fake = self.generator(noise)
                img_fake.append(fake[0])

        return img_fake
