import torch
import numpy as np


@torch.no_grad()
def sample_anneal_langevin(x, score_nn, sigmas, eps=2e-5, T=100):
    """
    Sampling procedure via annealed langevin dynamics
    :param x: (Tensor), [n_samples x C x W x H], initial samples
    :param score_nn: (nn.Module), score network
    :param T: (Int), number of steps for each level of noise
    :param sigmas: (Tensor), different levels of noise
    :param eps: (Float), step size
    :return: (Tensor), [n_samples x C x W x H]
    """
    samples = []

    for label, sigma in enumerate(sigmas):
        labels = torch.ones(size=(x.size(dim=0),), device=x.device) * label
        labels = labels.long()

        step_size = eps * (sigma / sigmas[-1]) ** 2

        for t in range(T):
            samples.append(torch.clamp(x, -1.0, 1.0).to('cpu'))

            z = torch.randn_like(x, device=x.device) * torch.tensor(torch.sqrt(step_size), device=x.device)

            score = score_nn(x, labels)
            x = x + step_size / 2 * score + z

    return samples
