import torch
import torch.nn as nn
from tqdm import tqdm


class NCSN(nn.Module):
    def __init__(self, score_nn, copy_score_nn, sampler, device, data_path, sigma_max):
        """
        :param score_nn: score network
        :param copy_score_nn: deep copy of the score network
        :param sampler: sampler model with corrector and predictor
        :param device: current working device
        :param data_path: path to dataset
        :param sigma_max: maximum value of the sigma (depend on the dataset)
        """
        super(NCSN, self).__init__()

        self.score_nn = score_nn
        self.copy_score_nn = copy_score_nn
        self.sampler = sampler

        self.device = device
        self.data_path = data_path

        self.sigma_max = sigma_max
        self.sigma_min = 0.01
        self.t_min = 1e-5
        self.t_max = 1

        self.T = 1000

    @torch.no_grad()
    def _sigma_calc(self, t):
        """
        Calculation function sigma(t), variance of the q(x(t)|x(0))
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

    @torch.no_grad()
    def p_sample(self, x):
        """
        Sampling from p_{theta}(x(0) | x(t))
        :param x: (Tensor) [b_size x C x W x H]
        :return: (Tensor) [b_size x C x W x H]
        """
        self.score_nn.eval()

        # Discretization of the continuous time
        time_steps = torch.linspace(self.t_min, self.t_max, self.T).to(self.device)

        for t in reversed(range(self.T - 1)):
            # Sigmas calculation
            t1 = time_steps[t + 1].repeat(x.size(0))
            t0 = time_steps[t].repeat(x.size(0))

            sigma_t1 = self._sigma_calc(t1)
            sigma_t = self._sigma_calc(t0)

            # Predictor step
            score = self.score_nn(x, t1)
            x = self.sampler.predictor_step(x, score, sigma_t1, sigma_t)

            # Corrector step
            score = self.score_nn(x, t0)
            x = self.sampler.corrector_step(x, score)

        self.score_nn.train()

        return x

    def loss(self, score_approx, score_true, sigma_t):
        """
        MSE loss for score matching
        :param score_approx: [b_size x C x W x H] approximation of the score
        :param score_true: [b_size x C x W x H] true score
        :param sigma_t: [b_size]
        :return: (Float)
        """
        loss = ((score_approx + score_true / sigma_t[:, None, None, None] ** 0.5) ** 2).mean(axis=(1, 2, 3))
        weighted_loss = loss * sigma_t ** 2
        return weighted_loss.mean()

    def fit(self, n_steps, trainloader):
        n_params = sum(p.numel() for p in self.score_nn.parameters())
        print(f'Number of parameters is {n_params}')

        opt = torch.optim.Adam(lr=2e-4, params=self.score_nn.parameters())

        for step in tqdm(range(n_steps)):
            x_0 = next(iter(trainloader))[0].to(self.device)
            size = x_0.size()

            t = torch.FloatTensor(size[0]).uniform_(self.t_min, self.t_max).to(self.device)
            sigma_t = self._sigma_calc(t)
            x_t, z = self.q_sample(x_0, sigma_t)

            score_approx = self.score_nn(x_t, t)
            score_true = z

            loss = self.loss(score_approx, score_true, sigma_t)

            print(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()
