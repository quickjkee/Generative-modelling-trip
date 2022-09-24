import torch
import torch.nn as nn


class Sampler(nn.Module):
    def __init__(self, score_nn, device):
        super(Sampler, self).__init__()

        self.score_nn = score_nn
        self.device = device

        self.r = 0.16

    def predictor_step(self, x, score, sigma_t1, sigma_t):
        """
        Calculating predictor step (Euler-Maruyama)
        :param x: (Tensor) [b_size x C x W x H]
        :param score: (Tensor) [b_size x C x W x H]
        :param sigma_t1: (Tensor) [b_size]
        :param sigma_t: (Tensor) [b_size]
        :return: (Tensor) [b_size x C x W x H]
        """
        z = torch.randn_like(x, device=x.device)

        sigma = (sigma_t1 ** 2 - sigma_t ** 2)[:, None, None, None]
        x = x + sigma * score + torch.sqrt(sigma) * z

        return x

    def corrector_step(self, x, score):
        """
        Corrector step (Langevin dynamics)
        :param x: (Tensor) [b_size x C x W x H]
        :param score: (Tensor) [b_size x C x W x H]
        :return: (Tensor) [b_size x C x W x H]
        """
        z = torch.randn_like(x, device=x.device)

        noise_norm = torch.norm(z.reshape(z.shape[0], -1), dim=-1).mean()
        score_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
        eps = (noise_norm / score_norm) ** 2 * 2 * self.r

        x = x + eps * score + torch.sqrt(2 * eps)[:, None, None, None] * z

        return x
