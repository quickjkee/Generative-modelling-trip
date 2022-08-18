import torch.nn as nn
import torch
import time
import os
import torch.nn.functional as F


class DDPM(nn.Module):
    def __init__(self, score_nn, device, data_path, T=1000, betas=None):
        """
        :param score_nn: (nn.Module), score networks
        :param betas: magnitudes of nosie
        :param T: (Int), number of noise levels
        :param device: current working device
        """
        super(DDPM, self).__init__()

        self.score_nn = score_nn
        self.device = device
        self.data_path = data_path

        if betas is None:
            self.betas = torch.linspace(1e-4, 0.02, steps=T).to(self.device)
        else:
            self.betas = betas.to(self.device)

        self.T = T
        self.alphas = 1 - self.betas

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

    def _checkpoint(self, i):
        dir = f'{self.data_path}/models_check'
        if not os.path.exists(dir):
            os.mkdir(dir)

        torch.save(self.score_nn.state_dict(), dir + f'/ddpm_iter{i}')

    @torch.no_grad()
    def sample(self, size):
        """
        Size of sample
        :param size: (Tuple), [n_samples x C x W x H]
        :return: (Tensor), [n_samples x C x W x H]
        """
        x = torch.randn(size).to(self.device)

        for t in reversed(range(self.T)):
            if t == 0:
                z = 0
            else:
                z = torch.randn(size).to(self.device)
            t = torch.tensor([t] * size[0])
            eps = self.score_nn(x, t)

            alpha = self.alphas[t][:, None, None, None]
            alpha_bar = self._alpha_bar(t)[:, None, None, None]

            x = 1 / torch.sqrt(alpha) * (x - (1 - alpha) / (torch.sqrt(1 - alpha_bar)) * eps) + torch.sqrt(
                1 - alpha) * z

        return x

    def fit(self, n_epochs, trainloader):
        opt = torch.optim.Adam(lr=2e-4, params=self.score_nn.parameters())

        for i in range(n_epochs):
            for j, batch in enumerate(trainloader):
                start_time = time.time()

                size = batch[0].size()

                x0 = batch[0].to(self.device)
                t = torch.randint(high=self.T, size=(size[0],))
                eps = torch.rand(size)

                eps_approx = self.forward(x0, t, eps)
                loss = self.loss(eps_approx, eps)

                opt.zero_grad()
                loss.backward()
                opt.step()

                end_time = time.time() - start_time

                print(f'Epoch {i}/{n_epochs}, Batch {j}/{len(trainloader)} \n'
                      f'Loss {round(loss.item(), 5)} \n'
                      f'Batch time {round(end_time, 5)}')

            self._checkpoint(i)
