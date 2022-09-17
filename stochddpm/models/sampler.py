import torch
import torch.nn as nn


class Sampler(nn.Module):
    def __init__(self, score_nn, device, sigma_max):
        super(Sampler, self).__init__()

        self.score_nn = score_nn
        self.device = device

        self.sigma_max = sigma_max
        self.sigma_min = 0.01

        self.N = 1000
        self.sigmas = torch.linspace()
        self.r = 0.16
