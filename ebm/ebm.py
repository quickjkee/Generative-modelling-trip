import torch.nn as nn
import torch
import time


class EBM(nn.Module):
    """
    Realization of the energy based model
    """

    def __init__(self, sampler, energy, device):
        """
        :param sampler: (Sampler), Current sampler (HMC or Langevin or any other)
        :param energy: (nn.Module), Backbone for approximation of the energy
        :param device: Current working device
        """
        super(EBM, self).__init__()

        self.sampler = sampler
        self.energy = energy

        self.device = device

        self.alpha = 1

    def _loss(self, out_real, out_model):
        """
        Calculation loss based on following formula

        L = out_real - out_model + (out_real**2 + out_model**2)
          = contrastive_div + regularization

        contrastive_div = out_real - out_model
        regularization = (out_real**2 + out_model**2)

        Contrastive divergence is main loss.
        Regularization is necessary to prevent high fluctuations

        :param out_real: (Tensor), [b_size x 1], Energy of real batch
        :param out_model: (Tensor), [b_size x 1], Energy of model batch
        :return: (Float), loss value
        """

        contrastive_div = out_real.mean() - out_model.mean()
        regularization = (out_real**2 + out_model**2).mean()

        loss = contrastive_div + self.alpha * regularization

        return loss, contrastive_div, regularization

    def sample(self, eval=False):
        """
        Function for sampling from EBM
        :return: (Tensor), [b_size x C x W x H]
        """
        samples = self.sampler.sample(self.energy)

        return samples

    def fit(self, trainloader, testloader, n_epochs):
        """
        Training procedure
        :param trainloader: (Dataloader), train data
        :param testloader: (Dataloader), test data
        :param n_epochs: (Integer), number of epochs
        :return:
        """

        params = list(self.energy.parameters())
        opt = torch.optim.Adam(lr=3e-4, params=params)

        print(f'opt={type(self.energy).__name__}, epochs={n_epochs}, device={self.device}')

        for i in range(n_epochs):

            for j, batch in enumerate(trainloader):
                start_time = time.time()

                opt.zero_grad()

                # Forward pass for real data
                batch_real = batch[0].to(self.device)
                out_real = self.energy(batch_real)

                # Forward pass for model data
                batch_model = self.sample().to(self.device)
                out_model = self.energy(batch_model)

                # Loss calculation
                loss, contrastive, regularization = self._loss(out_real, out_model)

                # Update params
                loss.backward()
                opt.step()

                stop_time = time.time() - start_time

                print(f'[Epoch {i}/{n_epochs}][Batch {j}/{len(trainloader)}] \n'
                      f'Contrastive loss {round(contrastive.item(), 6)} \n'
                      f'Regularization {round(regularization.item(), 6)} \n'
                      f'Batch time {round(stop_time, 3)} sec \n'
                      '======================================')
