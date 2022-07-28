import random

import torch
import numpy as np


class Sampler:
    """
    Realization of the sampler from unnormalized distribution based on Langevin Markov Chain
    """

    def __init__(self, n_samples, n_steps, step_size, device, img_shape=None):
        """
        :param img_shape: (Tuple), shape of image to sample
        :param n_samples: (Int), number of samples to create
        :param n_steps: (Int), number of steps in Langevin MCMC
        :param device: current working device
        :param step_size: (Int), step size in Langevin MCMC
        """
        super(Sampler, self).__init__()

        self._n_samples = n_samples
        self._step_size = step_size

        if img_shape is None:
            img_shape = (1, 32, 32)

        self.n_steps = n_steps
        self.img_shape = img_shape

        self.device = device

        # Buffer container to collect previous sample
        self._buffer = []

    def _langevin(self, e_model, inp_img):
        """
        Langevin MCMC
        :param e_model: (nn.Module), energy of the Boltzmann distribution (model distribution)
        :param inp_img: (Tensor), [n_samples x C x W x H], initial images
        :return: (Tuple(Tensor, Tensor)), updated images and momentum
        """
        inp_img = inp_img.to(self.device)

        # Change mode
        e_model.eval()

        # In HMC we only interested in gradient wrs to input
        for p in e_model.parameters():
            p.requires_grad = False
        inp_img.requires_grad = True

        tensor_noise = torch.empty((self._n_samples,) + self.img_shape)

        # Using Langevin dynamics n_steps times
        for _ in range(self.n_steps):
            noise = tensor_noise.normal_(mean=0, std=0.005).to(self.device)
            inp_img.data.add_(noise.data)
            inp_img.data.clamp_(min=-1.0, max=1.0)

            # Part 1. Gradient calculation
            out = -e_model(inp_img)
            out.sum().backward()
            inp_img.grad.data.clamp_(-0.03, 0.03)

            # Part 2. Images update
            inp_img.data.add_(-self._step_size * inp_img.grad.data)
            inp_img.grad.zero_()
            inp_img.data.clamp_(min=-1.0, max=1.0)

        for p in e_model.parameters():
            p.requires_grad = True
        e_model.train(True)

        inp_img.detach()

        return inp_img

    def sample(self, e_model, eval=False):
        """
        Sampling using Langevin MCMC
        :param e_model: (nn.Module), energy of the Boltzmann distribution (model distribution)
        :return: (Tensor), [_n_samples x C x W x H], samples from model distribution
        """
        if eval:
            inp_img = torch.rand((self._n_samples,) + self.img_shape) * 2 - 1

        else:
            # Initialization
            if len(self._buffer) == 0:
                inp_img = torch.rand((self._n_samples,) + self.img_shape) * 2 - 1
            else:
                n_new = np.random.binomial(self._n_samples, 0.05)
                rand_img = (torch.rand((n_new,) + self.img_shape) * 2 - 1).to(self.device)
                old_img = torch.cat(random.choices(self._buffer, k=self._n_samples - n_new), dim=0)
                inp_img = torch.cat([rand_img, old_img], dim=0).detach().to(self.device)

        # Langevin dynamics step
        out_img = self._langevin(e_model=e_model,
                                 inp_img=inp_img)

        self._buffer = list(out_img.chunk(self._n_samples, dim=0))

        return out_img
