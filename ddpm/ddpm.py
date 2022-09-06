import torch.nn as nn
import torch
import time
import os
import torch.nn.functional as F
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image
from utils.fid import fid
from tqdm import tqdm


def warmup_lr(step):
    return min(step, 5000) / 5000


class DDPM(nn.Module):
    def __init__(self, score_nn, copy_score_nn, device, data_path, n_eval, T=1000, betas=None):
        """
        :param score_nn: (nn.Module), score networks
        :param copy_score_nn: (nn.Module), copy of score to make EMA
        :param betas: magnitudes of noise
        :param data_path: (String), path to data with dataset
        :param T: (Int), number of noise levels
        :param device: current working device
        """
        super(DDPM, self).__init__()

        self.score_nn = score_nn.to(device)
        self.copy_score_nn = copy_score_nn.to(device)
        self.device = device
        self.n_eval = n_eval
        self.data_path = data_path

        if betas is None:
            self.betas = torch.linspace(1e-4, 0.02, steps=T).to(self.device)
        else:
            self.betas = betas.to(self.device)

        self.T = T
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

        self.n_batches = 10
        self.ema_decay = 0.9999

        if os.path.exists(f'{data_path}/fid_results/ddpm_fids.txt'):
            self._fids = list(np.loadtxt(f'{data_path}/fid_results/ddpm_fids.txt'))
        else:
            self._fids = []

    def forward(self, x0, t, eps):
        """
        Calculation of the score for real object, time label and noise
        :param x0: (Tensor), [b_size x C x W x H], real object
        :param eps: (Tensor), [b_size x C x W x H], noise
        :param t: (Tensor), [b_size], time labels
        :return: (Tensor), [b_size x C x W x H], score
        """
        alphas_bar = self.alphas_bar[t][:, None, None, None]
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

    @torch.no_grad()
    def _ema_update(self):
        """
        Updating weight of the copy model using EMA
        """
        source_dict = self.score_nn.state_dict()
        target_dict = self.copy_score_nn.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * self.ema_decay +
                source_dict[key].data * (1 - self.ema_decay))

    def _checkpoint(self, i):
        """
        Save current state of the model
        :param i: (Int) number of epoch
        """
        dir = f'{self.data_path}/models_check'
        if not os.path.exists(dir):
            os.mkdir(dir)

        torch.save(self.copy_score_nn.state_dict(), dir + f'/ddpm_iter{i}.pkl')

    def _save_samples(self, i, loader):
        """
        Save one batch sample
        :param i: (Int) number of epoch
        :param loader: (DataLoader)
        """
        dir = f'{self.data_path}/eval_samples'
        if not os.path.exists(dir):
            os.mkdir(dir)

        samples = next(iter(loader))[0]
        img = samples.to('cpu')
        save_image(img.float(), "{}/{}.png".format(dir, i))

    @torch.no_grad()
    def _calculate_fid(self, i, dataloader, size, n_batches):
        """
        Calculation FID
        :param dataloader: (nn.Dataloder), true data
        :param size: (Tuple), size of data
        :param n_batches: (Int), number of batches to calculate
        :return: (Float), calculated FID score
        """
        z = torch.randn(size).to(self.device)

        # True data
        trueloader = dataloader

        # Creating test loader
        testloader = DataLoader(TensorDataset(self.sample(z, n_batches)), batch_size=size[0])
        # Save batch to validate
        self._save_samples(i, testloader)

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
        for t in tqdm(reversed(range(self.T))):
            if t == 0:
                z = 0
            else:
                z = torch.randn_like(x).to(self.device)

            t = torch.tensor([t] * x.size(0)).to(self.device)
            eps = self.copy_score_nn(x, t)

            alpha = self.alphas[t][:, None, None, None]
            alpha_bar = self.alphas_bar[t][:, None, None, None]

            x = 1 / torch.sqrt(alpha) * (x - (1 - alpha) / (torch.sqrt(1 - alpha_bar)) * eps) + torch.sqrt(
                1 - alpha) * z

        return (torch.clip(x, -1.0, 1.0) + 1) / 2

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

    def fit(self, n_steps, trainloader):
        n_params = sum(p.numel() for p in self.score_nn.parameters())
        print(f'Number of parameters is {n_params}')

        opt = torch.optim.Adam(lr=2e-4, params=self.score_nn.parameters())
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=warmup_lr)

        for step in tqdm(range(n_steps)):

            batch = next(iter(trainloader))[0].to(self.device)
            size = batch.size()

            t = torch.randint(high=self.T, size=(size[0],)).to(self.device)
            eps = torch.randn(size).to(self.device)

            eps_approx = self.forward(batch, t, eps)
            loss = self.loss(eps_approx, eps)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.score_nn.parameters(), 1.)

            opt.step()
            sched.step()

            self._ema_update()

            if step % (len(trainloader)) == 0:
                print(f'Loss {round(loss.item(), 5)}', flush=True)

            if step % self.n_eval == 0:
                self._checkpoint(step)
                self._calculate_fid(step, trainloader, size, self.n_batches)
