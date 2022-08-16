import torch.nn as nn
import torch
import time
import os


class NCSN(nn.Module):
    """
    Realization of the Score Model with
    - neural network for estimation of the score
    - sampler
    - noise levels
    """

    def __init__(self, score_nn, sampler, device, data_path, sigmas=None):
        """
        :param score_nn: (nn.Module), Neural network for approximation the score
        :param sampler: (Sampler), object with .sample() method
        :param sigmas: (Tensor), array of different noise levels
        :param device: current working device
        :param data_path: path to data with model
        :param lambda: (Tensor), weights in objective loss
        """
        super(NCSN, self).__init__()

        if sigmas is None:
            self.sigmas = torch.linspace(1, 0.01, steps=10).to(device)
        else:
            self.sigmas = sigmas.to(device)

        self.score_nn = score_nn.to(device)
        self.sampler = sampler

        self.device = device

        self.data_path = data_path

        self.used_sigma = None
        self.noise = None

    def _checkpoint(self, i):
        dir = f'{self.data_path}/models_check'
        if not os.path.exists(dir):
            os.mkdir(dir)

        torch.save(self.score_nn.state_dict(), dir + f'/ncsn_v2_iter{i}')

    def _dsm_loss(self, score):
        """
        Calculating DSM loss based on formula

        0.5 * |E_{ |N(z|0, I)p_{data}(x) } [|| score(x + sigma * z, sigma) + 1/sigma * z ||^2 ]

        :param score: (Tensor), [b_size x C x W x H], estimation of the score
        :return: (Tensor), [b_size] dsm loss
        """
        loss = 0.5 * torch.sum((score + self.noise / self.used_sigma) ** 2, dim=(1, 2, 3))

        return loss

    def _ssm_loss(self, score):
        pass

    def loss(self, score, type='dsm'):
        """
        Calculating loss based on formula

        mean( lambda * l(x | theta, sigma) ), l = l_ssm or l_dsm

        :param score: (Tensor), [b_size x C x W x H], estimation of the score
        :param type: (String): type of l
        :return: (Float): loss
        """

        if type == 'dsm':
            l = self._dsm_loss(score)
        else:
            l = self._ssm_loss(score)

        final_loss = torch.mean(l * self.used_sigma.squeeze() ** 2)

        return final_loss

    def forward(self, x):
        """
        Estimation of the score for different noise level
        Here we calculate one level of noise for each object.
        It is possible to calculate different levels of noise for each object
        Then output will be [b_size x L x C x W x H]
        :param x: (Tensor), [b_size x C x W x H], input object
        :return: (Tensor), [b_size x C x W x H], score, each object has his own noise
        """
        b_size = x.size(dim=0)
        labels = torch.randint(low=0, high=len(self.sigmas), size=(b_size,)).to(self.device)
        self.used_sigma = self.sigmas[labels].view(b_size, 1, 1, 1)
        self.noise = torch.randn_like(x).to(self.device)

        x_noised = x + self.noise * self.used_sigma
        score = self.score_nn(x_noised, self.used_sigma)

        return score

    def sample(self, size):
        x = torch.randn(size=size).to(self.device)
        samples = self.sampler(x, self.score_nn, self.sigmas)

        return samples

    def fit(self, n_epochs, trainloader, type='dsm'):
        """
        :param n_epochs: (Int), number of epochs
        :param trainloader: (Dataloader)
        :param type: (String), type of loss (dsm or ssm)
        :return:
        """
        params = self.score_nn.parameters()
        opt = torch.optim.Adam(lr=1e-4, params=params)

        for i in range(n_epochs):
            for j, batch in enumerate(trainloader):
                start_time = time.time()

                opt.zero_grad()

                batch = batch[0].to(self.device)

                score = self.forward(batch)
                loss = self.loss(score, type=type)

                loss.backward()
                opt.step()

                end_time = time.time() - start_time

                print(f'Epoch [{i}/{n_epochs}], batch [{j}/{len(trainloader)}] \n'
                      f'Loss {type} = {round(loss.item(), 4)} \n'
                      f'Time for batch {round(end_time, 3)}')

            self._checkpoint(i)
