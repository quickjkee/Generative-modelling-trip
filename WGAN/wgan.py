import torch.nn as nn
import torch
import time


class WGAN(nn.Module):
    """
    Wasserstein GAN model realization
    """

    def __init__(self, critic, generator, h_dim, device):
        """
        :param critic: (nn.Module), critic model
        :param generator: (nn.Module), generator model
        :param h_dim: (Integer), dimension of the latent variable
        :param device: current working device
        """
        super(WGAN, self).__init__()

        self.critic = critic
        self.generator = generator

        self.h_dim = h_dim

        self.device = device

    def _wasserstein_approximation(self, real_critic, fake_critic):
        """
        Approximation of the Wasserstein distance.
        Real Wasserstein metric is

        W(P, Q) = max_{f} [|E_{P} f(X) - |E_{Q} f(X)]

        Our approximation is

        W(P, Q) ~ [|E_{P(X)} critic(X) - |E_{P(Z)} critic(g(z))]

        :param real_critic: (Tensor), [b_size x 1], critic calculation for real samples
        :param fake_critic: (Tensor), [b_size x 1], critic calculation for fake samples
        :return: (Tensor), [1], approximation of the Wasserstein distance
        """

        # We put minus because the optimizer working in minimizing mode as standard
        approx = -(torch.mean(real_critic, dim=0) - torch.mean(fake_critic, dim=0))

        return approx

    def _generator_loss(self, fake_critic):
        """
        Calculating generator loss corresponding to the formula

        -|E_{P(z)} critic(g(z))

        :param fake_critic: (Tensor), [b_size x 1], critic calculation for fake samples
        :return: (Tensor), [1], loss
        """
        loss = -torch.mean(fake_critic)

        return loss

    def _critic_pass(self, x, z):
        """
        The critic forward pass and loss calculation
        :param x: (Tensor), [b_size x C x W x H]
        :param z: (Tensor), [b_size x h_dim]
        :return: (Tensor), [1]
        """

        #######################
        # FORWARD PASS
        # #####################

        # Calculating critic for real samples
        real_critic = self.critic.forward(x)

        # Calculating critic for fake samples
        fake_samples = self.generator.forward(z)
        fake_critic = self.critic.forward(fake_samples)

        #######################
        # LOSS CALCULATION
        # #####################

        loss = self._wasserstein_approximation(real_critic, fake_critic)

        return loss

    def _generator_pass(self, z):
        """
        The generator forward pass and loss calculation
        :param z: (Tensor), [b_size x h_dim]
        :return: (Tensor), [1], generator loss
        """

        #######################
        # FORWARD PASS
        # #####################

        # Calculating critic for fake samples
        fake_samples = self.generator.forward(z)
        fake_critic = self.critic.forward(fake_samples)

        #######################
        # LOSS CALCULATION
        # #####################

        loss = self._generator_loss(fake_critic)

        return loss

    def sample(self, z):
        """
        Sampling from model distribution
        :param z: (Tensor), [b_size x h_dim]
        :return: (Tensor), [b_size x C x W x H]
        """
        samples = self.generator.forward(z)

        return samples

    def fit(self, trainloader, testloader, b_size, n_epochs):
        """
        Training procedure
        :param trainloader: (Dataloader), train data
        :param testloader: (Dataloader), test data
        :param n_epochs: (Integer), number of epochs
        :return:
        """
        generator_params = list(self.generator.parameters())
        critic_params = list(self.critic.parameters())
        params = generator_params + critic_params

        opt = torch.optim.RMSprop(lr=5e-5, params=params)

        print(f'opt={type(opt).__name__}, epochs={n_epochs}, device={self.device}')

        for epoch in range(n_epochs):

            for i in range(len(trainloader)):

                #######################
                # CRITIC TRAINING
                # #####################

                start_time = time.time()

                critic_error = 0.0
                for _ in range(5):
                    opt.zero_grad()

                    # Real samples
                    batch_sample = next(iter(trainloader))[0].to(self.device)

                    # Fake samples
                    latent_sample = torch.randn(b_size, self.h_dim).to(self.device)

                    # Loss calculation
                    critic_loss = self._critic_pass(batch_sample, latent_sample)

                    # Backward and optimizer step
                    critic_loss.backward(inputs=critic_params)
                    opt.step()

                    critic_error += critic_loss.item()

                #######################
                # GENERATOR TRAINING
                #######################

                opt.zero_grad()

                # Fake samples
                latent_sample = torch.randn(b_size, self.h_dim).to(self.device)

                # Loss calculation
                gen_loss = self._generator_pass(latent_sample)

                # Backward and optimizer step
                gen_loss.backward(inputs=generator_params)
                opt.step()

                end_time = time.time() - start_time

                print(f'[Epoch {epoch}/{n_epochs}][Batch {i}/{len(trainloader)}] \n'
                      f'Generator loss {round(gen_loss.item(), 3)} \n'
                      f'Critic loss {round(critic_error / 5, 3)} \n'
                      f'Work time {round(end_time, 3)} sec \n'
                      )
