import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, noise_dim, out_dim):
        super(Generator, self).__init__()

        self.out_dim = out_dim
        self.noise_dim = noise_dim

        self.model = self._make_model()

    def _make_model(self):
        model = nn.Sequential(
            nn.ConvTranspose2d(self.noise_dim,
                               1024,
                               kernel_size=4,
                               stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024,
                               512,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512,
                               256,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,
                               128,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128,
                               1,
                               kernel_size=3,
                               stride=1,
                               padding=1),
            nn.Tanh(),
        )

        return model

    def forward(self, x):
        out = self.model(x)

        return out


class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super(Discriminator, self).__init__()

        self.in_dim = in_dim

        self.model = self._make_model()

    def _make_model(self):
        model = nn.Sequential(
            nn.Conv2d(self.in_dim, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        return model

    def forward(self, x):
        out = self.model(x)

        return out


class GAN(nn.Module):
    def __init__(self, in_dim, z_dim, device):
        super(GAN, self).__init__()

        self.z_dim = z_dim

        self.bce = torch.nn.BCELoss()

        self.disc = Discriminator(in_dim).to(device)
        self.gen = Generator(z_dim, in_dim).to(device)

        self.device = device

    def forward(self, noise):
        samples = self.gen(noise)

        return samples

    def _disc_loss(self, pred_true, pred_fake):
        targets_true = torch.full(size=(pred_true.size(dim=0), 1), fill_value=1, dtype=torch.float32).to(self.device)
        targets_fake = torch.full(size=(pred_true.size(dim=0), 1), fill_value=0, dtype=torch.float32).to(self.device)

        bce_true = self.bce(pred_true, targets_true)
        bce_fake = self.bce(pred_fake, targets_fake)

        bce = (bce_fake + bce_true) / 2

        return bce

    def _gen_loss(self, pred_fake):
        targets_fake = torch.full(size=(pred_fake.size(dim=0), 1), fill_value=1, dtype=torch.float32).to(self.device)
        bce_generator = self.bce(pred_fake, targets_fake)

        return bce_generator

    def fit(self, trainlaoder, n_epochs):
        g_opt = torch.optim.Adam(lr=2e-4, params=self.gen.parameters())
        d_opt = torch.optim.Adam(lr=2e-4, params=self.disc.parameters())

        for i in range(n_epochs):
            for j, batch in enumerate(trainlaoder):
                g_opt.zero_grad()
                d_opt.zero_grad()

                batch_real = batch[0].to(self.device)

                noise = torch.randn(batch_real.size(dim=0), self.z_dim, 1, 1).to(self.device)
                batch_fake = self.gen(noise)

                #####################
                # discriminator upd
                #####################
                d_real = self.disc(batch_real).view(-1, 1)
                d_fake = self.disc(batch_fake).view(-1, 1)

                d_loss = self._disc_loss(d_real, d_fake)
                d_loss.backward(retain_graph=True)

                d_opt.step()

                #####################
                # generator upd
                #####################
                d_fake = self.disc(batch_fake).view(-1, 1)
                g_loss = self._gen_loss(d_fake)
                g_loss.backward()

                g_opt.step()


                print(f'Epoch {i}/{n_epochs}, batch {j}/{len(trainlaoder)} \n'
                      f'Loss D {d_loss.item()} \n'
                      f'Loss G {g_loss.item()}')

