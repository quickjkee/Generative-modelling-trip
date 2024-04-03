import torch.nn as nn
import torch
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Began(nn.Module):
    def __init__(self, gen, ae, z_dim, device):
        """
        :param gen: (nn.Module), generator
        :param ae: (nn.Module), autoencoder
        :param z_dim: (Int), dimension of latent vector
        :param device: current working device
        """
        super(Began, self).__init__()

        self.gen = gen.to(device)
        self.ae = ae.to(device)

        self.k = 0
        self.gamma = 1
        self.lam = 0.1

        self.z_dim = z_dim

        self.device = device

    @torch.no_grad()
    def sample(self, noise):
        """
        Sampling from began model
        :param noise: (Tensor), [b_size, z_dim]
        :return: (Tensor), [b_size x C x W x H]
        """
        if noise.size(dim=1) != self.z_dim:
            raise ValueError("Input size of latent vector does not matches with initialized")

        samples = self.gen(noise)

        return samples

    def _ae_loss(self, real_batch, ae_real, fake_batch, ae_fake):
        ae_real_loss = F.l1_loss(real_batch, ae_real)
        ae_fake_loss = F.l1_loss(fake_batch, ae_fake)

        ae_loss = ae_real_loss - self.k * ae_fake_loss

        return ae_loss, ae_real_loss, ae_fake_loss

    @torch.no_grad()
    def _k_update(self, ae_real_loss, ae_fake_loss):
        balance = (self.gamma * ae_real_loss - ae_fake_loss).data
        self.k = torch.clamp(self.k + self.lam * balance, 0, 1).item()

    @torch.no_grad()
    def measure_loss(self, real_batch, ae_real, fake_batch, ae_fake):
        first_term = F.l1_loss(real_batch, ae_real)
        second_term = torch.abs(self.gamma * first_term - F.l1_loss(fake_batch, ae_fake))

        return first_term + second_term

    def _gen_loss(self, fake_batch, ae_fake):
        ae_fake_loss = F.l1_loss(fake_batch, ae_fake)

        return ae_fake_loss

    def fit(self, trainloader, n_epochs):
        g_opt = torch.optim.Adam(lr=1e-4, betas=(0.5, 0.999), params=self.gen.parameters())
        ae_opt = torch.optim.Adam(lr=1e-4, betas=(0.5, 0.999), params=self.ae.parameters())

        G_scheduler = ReduceLROnPlateau(g_opt, factor=0.5, threshold=0.01,
                                        patience=5 * len(trainloader))
        D_scheduler = ReduceLROnPlateau(ae_opt, factor=0.5, threshold=0.01,
                                        patience=5 * len(trainloader))

        for i in range(n_epochs):

            for j, batch in enumerate(trainloader):
                start_time = time.time()

                real_batch = batch[0].to(self.device)
                b_size = real_batch.size(dim=0)

                noise = torch.randn(b_size, self.z_dim).to(self.device)
                fake_batch = self.gen(noise)

                ##################################
                # Auto Encoder update
                ##################################
                ae_opt.zero_grad()

                ae_real = self.ae(real_batch)
                ae_fake = self.ae(fake_batch)

                ae_loss, ae_real_loss, ae_fake_loss = self._ae_loss(real_batch,
                                                                    ae_real,
                                                                    fake_batch,
                                                                    ae_fake)

                ae_loss.backward()
                ae_opt.step()

                ##################################
                # Generator update
                ##################################
                g_opt.zero_grad()

                noise = torch.randn(b_size, self.z_dim).to(self.device)
                fake_batch = self.gen(noise)
                ae_fake = self.ae(fake_batch)

                gen_loss = self._gen_loss(fake_batch,
                                          ae_fake)

                gen_loss.backward()
                g_opt.step()

                ##################################
                # Validation
                ##################################
                self._k_update(ae_real_loss, ae_fake_loss)

                measure_loss = self.measure_loss(real_batch,
                                                 ae_real,
                                                 fake_batch,
                                                 ae_fake)

                stop_time = time.time() - start_time

                print(f'Epoch [{i}/{n_epochs}], batch [{j}/{len(trainloader)}] \n'
                      f'Loss {round(measure_loss.item(), 4)} \n'
                      f'Batch time {round(stop_time, 5)}')

            D_scheduler.step(measure_loss)
            G_scheduler.step(measure_loss)
