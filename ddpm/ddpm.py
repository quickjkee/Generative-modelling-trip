import torch.nn as nn
import torch
import time
import os
import torch.nn.functional as F
import numpy as np

from torch.utils.data import TensorDataset, DataLoader, Subset
from utils.fid import fid


class DDPM(nn.Module):
    def __init__(self, score_nn, device, data_path, T=1000, betas=None):
        """
        :param score_nn: (nn.Module), score networks
        :param betas: magnitudes of nosie
        :param T: (Int), number of noise levels
        :param device: current working device
        """
        super(DDPM, self).__init__()

        self.score_nn = score_nn.to(device)
        self.device = device
        self.data_path = data_path

        if betas is None:
            self.betas = torch.linspace(1e-4, 0.02, steps=T).to(self.device)
        else:
            self.betas = betas.to(self.device)

        self.T = T
        self.alphas = 1 - self.betas

        self.n_batches = 10
        self._fids = []

    @torch.no_grad()
    def _alpha_bar(self, t):
        """
        Calculation alpha with bar
        :param t: (Tensor), [b_size], time labels
        :return: (Tensor), [b_size]
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t)

        return torch.stack([torch.prod(self.alphas[:idx + 1]) for idx in t])

    def forward(self, x0, t, eps):
        """
        Calculation of the score for real object, time label and noise
        :param x0: (Tensor), [b_size x C x W x H], real object
        :param eps: (Tensor), [b_size x C x W x H], noise
        :param t: (Tensor), [b_size], time labels
        :return: (Tensor), [b_size x C x W x H], score
        """
        alphas_bar = self._alpha_bar(t)[:, None, None, None]
        inp = torch.sqrt(alphas_bar) * x0 + torch.sqrt(1 - alphas_bar) * eps
        out = self.score_nn(inp, t)

        return out

    def loss(self, score, eps):
        """
        :param score: (Tensor), [b_size x C x W x H]
        :param eps: (Tensor), [b_size x C x W x H]
        :return: (Float)
        """
        loss = F.mse_loss(score, eps)

        return loss

    def _checkpoint(self):
        dir = f'{self.data_path}/models_check'
        if not os.path.exists(dir):
            os.mkdir(dir)

        torch.save(self.score_nn.state_dict(), dir + f'/ddpm_iter{0}')

    @torch.no_grad()
    def _calculate_fid(self, dataloader, size, n_batches):
        """
        Calculation FID
        :param dataloader: (nn.Dataloder), true data
        :param size: (Tuple), size of data
        :param n_batches: (Int), number of batches to calculate
        :return: (Float), calculated FID score
        """
        z = torch.randn(size).to(self.device)

        # Creating sub loader from dataloader to calculate fid
        true_dataset = dataloader.dataset
        indices = np.arange(len(true_dataset))[:n_batches * size[0]]
        np.random.shuffle(indices)

        true_dataset_part = Subset(true_dataset, indices)
        trueloader = DataLoader(true_dataset_part, batch_size=size[0])

        # Creating test loader
        testloader = DataLoader(TensorDataset(self.sample(z, n_batches)), batch_size=size[0])

        fid_ = fid(trueloader, testloader, size[0], self.device)
        self._fids.append(round(fid_, 3))

        print(f'Calculated fids {self._fids}')

        dir = f'{self.data_path}/fid_results'
        if not os.path.exists(dir):
            os.mkdir(dir)

        np.savetxt(dir + f'/ddpm_fids.txt', np.array(self._fids), delimiter=',')

        return fid_

    # Sampling for one batch
    @torch.no_grad()
    def batch_sample(self, x):
        for t in reversed(range(self.T)):
            if t == 0:
                z = 0
            else:
                z = torch.randn_like(x).to(self.device)
                t = torch.tensor([t] * x.size(0)).to(self.device)
                eps = self.score_nn(x, t)

                alpha = self.alphas[t][:, None, None, None]
                alpha_bar = self._alpha_bar(t)[:, None, None, None]

                x = 1 / torch.sqrt(alpha) * (x - (1 - alpha) / (torch.sqrt(1 - alpha_bar)) * eps) + torch.sqrt(
                    1 - alpha) * z

        return torch.clip(x, -1.0, 1.0)

    @torch.no_grad()
    def sample(self, x, n_batches):
        """
        Size of sample
        :param x: (Tensor), [b_size x C x W x H], random input object
        :return: (Tensor), [n_batches * b_size x C x W x H]
        """
        x = x.to(self.device)

        samples = None
        for _ in range(n_batches):
            out = self.batch_sample(x)

            if samples is None:
                samples = out
            else:
                samples = torch.cat((samples, out), dim=0)

        return samples

    def fit(self, n_epochs, trainloader):
        n_params = sum(p.numel() for p in self.score_nn.parameters())
        print(f'Number of parameters is {n_params}')

        opt = torch.optim.Adam(lr=2e-4, params=self.score_nn.parameters())

        for i in range(n_epochs):
            for j, batch in enumerate(trainloader):
                start_time = time.time()

                size = batch[0].size()

                x0 = batch[0].to(self.device)
                t = torch.randint(high=self.T, size=(size[0],)).to(self.device)
                eps = torch.randn(size).to(self.device)

                eps_approx = self.forward(x0, t, eps)
                loss = self.loss(eps_approx, eps)

                opt.zero_grad()
                loss.backward()
                opt.step()

                end_time = time.time() - start_time

                print(f'Epoch {i}/{n_epochs}, Batch {j}/{len(trainloader)} \n'
                      f'Loss {round(loss.item(), 5)} \n'
                      f'Batch time {round(end_time, 5)}')

            self._checkpoint()

            if i % 20 == 0:
                self._calculate_fid(trainloader, batch[0].size(), self.n_batches)
