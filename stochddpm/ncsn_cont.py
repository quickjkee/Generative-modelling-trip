import torch
import torch.nn as nn
from tqdm import tqdm


class NCSN(nn.Module):
    def __init__(self, score_nn, copy_score_nn, device, data_path, sigma_max):
        """
        :param score_nn: score network
        :param copy_score_nn: deep copy of the score network
        :param device: current working device
        :param data_path: path to dataset
        :param sigma_max: maximum value of the sigma (depend on the dataset)
        """
        super(NCSN, self).__init__()

        self.score_nn = score_nn
        self.copy_score_nn = copy_score_nn

        self.device = device
        self.data_path = data_path

        self.sigma_max = sigma_max
        self.sigma_min = 0.01
        self.t_min = 1e-5
        self.t_max = 1

    @torch.no_grad()
    def _sigma_calc(self, t):
        """
        Calculation function sigma(t)
        :param t: (Tensor), [b_size]
        :return: (Tensor), [b_size]
        """
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t

        return sigma

    @torch.no_grad()
    def q_sample(self, x_0, sigma_t):
        """
        Sampling from q(x(t) | x(0), t)
        :param sigma_t: (Tensor), [b_size]
        :param x_0: (Tensor), [b_size x C x W x H], batch of real objects
        :return: (Tensor), [b_size x C x W x H]
        """

        z = torch.randn(x_0.size(), device=x_0.device)
        x_t = x_0 + sigma_t[:, None, None, None] ** 0.5 * z

        return x_t, z

    def loss(self, score_approx, score_true, sigma_t):
        """
        MSE loss for score matching
        :param score_approx: [b_size x C x W x H] approximation of the score
        :param score_true: [b_size x C x W x H] true score
        :return: (Float)
        """
        loss = torch.sum((score_approx * sigma_t[:, None, None, None] ** 0.5 + score_true) ** 2, dim=(1, 2, 3)).mean()
        return loss

    def fit(self, n_steps, trainloader):
        n_params = sum(p.numel() for p in self.score_nn.parameters())
        print(f'Number of parameters is {n_params}')

        opt = torch.optim.Adam(lr=2e-4, params=self.score_nn.parameters())

        for step in tqdm(n_steps):
            x_0 = next(iter(trainloader))[0].to(self.device)
            size = x_0.size()

            t = torch.FloatTensor(size[0]).uniform_(self.t_min, self.t_max).to(self.device)
            sigma_t = self._sigma_calc(t)
            x_t, z = self.q_sample(x_0, sigma_t)

            score_approx = self.score_nn(x_t, t)
            score_true = z

            loss = self.loss(score_approx, score_true, sigma_t)

            opt.zero_grad()
            loss.backward()
            opt.step()
