import torch
import time

from torch import nn
from torch.autograd import Variable


class VAE_GAN(nn.Module):
    def __init__(self, Encoder, Decoder, Discriminator, hidden_dim, conv_dims, device):
        """
        Creating VAE_GAN model based on following criterion

        L = L_prior + L_gan + L_l

        :param Encoder: Encoder model
        :param Decoder: Decoder model
        :param Discriminator: Discriminator model
        :param hidden_dim: (Int) Dimension of hidden space
        :param conv_dims: (Int) List of channels dimensional through conv layers
        :param device: Current working device
        """
        super(VAE_GAN, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(hidden_dim, conv_dims, device).to(device)
        self.decoder = Decoder(hidden_dim, conv_dims, device).to(device)
        self.discriminator = Discriminator(hidden_dim, conv_dims, device).to(device)

        self.bce = nn.BCELoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')

        self.rate = 1
        self.gamma = 1e-3

    def __call__(self, x):
        """
        Only the encoder and decoder is used for the call
        Sampling from latent distribution q(z|x) and
        reconstructing with decoder
        :param x: (Tensor) [B x C x W x H]
        :return: (Tensor) Similar as input object
        """
        mu, sigma = self.encoder(x)
        latent_sample = self.encoder.sample(mu, sigma)
        object_sample = self.decoder.forward(latent_sample)
        probab = self.discriminator(object_sample)

        return probab

    def prior_loss(self, mu, log_sigma):
        """
        Calculating KL-divergence between q(z|x) and N(0, I)
        :param mu: (Tensor) [B x h_dim] Expectation of latent conditional distribution
        :param log_sigma: (Tensor) [B x h_dim] Log variance of latent conditional distribution
        :return: (Float) Value of KL divergence
        """
        loss = (-0.5 * (1 + log_sigma - torch.exp(log_sigma) - mu ** 2)) \
            .view(mu.size(dim=0), -1).sum(dim=1).mean(dim=0)

        return loss

    def gan_loss(self, dis_x, dis_x_latent, dis_x_noise):
        """
        Calculating gan_loss which is equal

        L_gan = binary_ce(D(x), ones) + binary_ce(D(x_de), zeros) + binary_ce(D(x_noise), zeros),

        ## x - real objects
        ## x_de = G(E(x)) - decoded samples from latent distribution q(z|x)
        ## x_noise = G(noise) - decoded samples from noise p(z) = N(0, I)
        ## binary_ce - binary cross entropy

        :param dis_x: (Tensor) [B x 1] D(x) in previous notation
        :param dis_x_latent: (Tensor) [B x 1] D(x_de)
        :param dis_x_noise: (Tensor) [B x 1] D(x_noise)
        :return: (Float) Value of GAN loss
        """
        batch_size = dis_x.size(dim=0)

        ones = torch.ones(size=(batch_size, 1)).to(self.device)
        zeros = torch.zeros(size=(batch_size, 1)).to(self.device)

        gan_loss_dis = (self.bce(dis_x, ones) + self.bce(dis_x_latent, zeros) + self.bce(dis_x_noise, zeros)).mean(dim=0) / 3
        gan_loss_gen = (self.bce(dis_x_latent, ones) + self.bce(dis_x_noise, ones)).mean(dim=0) / 2

        return gan_loss_dis, gan_loss_gen

    def hidden_loss(self, d_l_x, d_l_x_de):
        """
        Calculating reconstruction loss with following formula
        using reparameterizarion trick and M-C estimation

        -|E_q [ log{p(D_l(x)|z)} ] ~ -log{p(D_l(x)|z)}

        ## D_l(x) - output of the l-th layer of the discriminator
        ## p(D_l(x)|z) = N(D_l(x) | D_l(x_de), I) => -log{p(D_l(x)|z)} = +||D_l(x) - D_l(x_de)||^2
        ## x_de = G(E(x)) - decoded samples from latent distribution q(z|x)
        ## E(x) means sample from q(z|x)

        :param d_l_x: (Tensor) [B x 1] D_l(x) in previous notation
        :param d_l_x_de: (Tensor) [B x 1] D_l(x_de)
        :return: (Float) Value of hidden discriminator loss
        """

        hidden_loss = 0.5 * self.mse(d_l_x, d_l_x_de).sum(dim=1).mean(dim=0)

        return hidden_loss

    def get_losses(self, x):
        """
        Calculating losses for encoder, generator, discriminator
        :param x: (Tensor) [B x C x W x H]
        :return: List(Float) Three losses
        """
        batch_size = x.size(dim=0)

        ############ FORWARD PASS ############

        mu, log_sigma = self.encoder(x)
        z_latent = self.encoder.sample(mu, log_sigma)
        x_latent = self.decoder(z_latent)

        # Sampling from prior p(z)
        z_noise = torch.randn(size=(batch_size, self.hidden_dim)).to(self.device)
        x_noise = self.decoder(z_noise)

        dis_x_latent, dis_feature_latent = self.discriminator(x_latent, is_loss=True)
        dis_x_noise = self.discriminator(x_noise)
        dis_x, dis_feature_real = self.discriminator(x, is_loss=True)

        ############ LOSSES CALCULATION ############

        ########
        # Calculating encoder loss based on formula
        #      L_encoder = L_prior + L_l
        ########

        l_prior = self.prior_loss(mu, log_sigma)
        l_l = self.hidden_loss(dis_feature_real, dis_feature_latent)

        l_encoder = l_prior + l_l

        ########
        # Calculating generator loss based on formula
        #      L_gan = -L_gan + L_l
        ########

        l_gan_dis, l_gan_gen = self.gan_loss(dis_x_latent, dis_x_noise, dis_x)

        l_gen = (self.gamma * l_l + l_gan_gen)

        ########
        # Calculating discriminator loss based on formula
        #      L_dis = L_gan
        ########

        l_dis = l_gan_dis

        return l_encoder, l_gen, l_dis

    def fit(self, trainloader, testloader, epochs):
        """
        Optimizing VAE/GAN model
        :param trainloader: (Dataloader) Train dataloader
        :param testloader: (Dataloader) Test dataloader
        :param epochs: (Int) Number of epochs
        :return: (dict) History of losses
        """
        en_optim = torch.optim.Adam(params=self.encoder.parameters(), lr=3e-4)
        gen_optim = torch.optim.Adam(params=self.decoder.parameters(), lr=3e-4)
        dis_optim = torch.optim.Adam(params=self.discriminator.parameters(), lr=3e-4)

        print('opt=%s(lr=%f), epochs=%d, device=%s,\n'
              'en loss \u2193, gen loss \u2193, dis loss \u2191' % \
              (type(en_optim).__name__,
               en_optim.param_groups[0]['lr'], epochs, self.device))

        history = {}
        history['loss'] = []
        history['val_loss'] = []

        train_len = len(trainloader.dataset)
        test_len = len(testloader.dataset)

        for epoch in range(epochs):
            start_time = time.time()

            ############ TRAINING PART ############

            for i, batch_samples in enumerate(trainloader):
                batch_samples = Variable(batch_samples[0], requires_grad=False).to(self.device)

                loss_en, loss_gen, loss_dis = self.get_losses(batch_samples)

                ##############
                # ENCODER FIT
                ##############
                en_optim.zero_grad()

                loss_en.backward(retain_graph=True, inputs=list(self.encoder.parameters()))
                en_optim.step()

                ##############
                # GENERATOR FIT
                ##############
                gen_optim.zero_grad()

                loss_gen.backward(retain_graph=True, inputs=list(self.decoder.parameters()))
                gen_optim.step()

                ##############
                # DISCRIMINATOR FIT
                ##############
                dis_optim.zero_grad()

                loss_dis.backward(inputs=list(self.discriminator.parameters()))
                dis_optim.step()

                end_time = time.time()
                work_time = end_time - start_time

                print('Epoch/batch %3d/%3d \n'
                      'train_en_loss %5.5f, train_gen_loss %5.5f, train_dis_loss %5.5f \n'
                      'batch time %5.2f sec' % \
                      (epoch + 1, i,
                       loss_en.item(), loss_gen.item(), loss_dis.item(),
                       work_time))

            ############ VALIDATION PART ############

            val_en_loss = 0.0
            val_gen_loss = 0.0
            val_dis_loss = 0.0

            with torch.no_grad():
                for i, batch_samples in enumerate(testloader):
                    batch_samples = batch_samples[0].to(self.device)

                    loss_en, loss_gen, loss_dis = self.get_losses(batch_samples)

                    val_en_loss += loss_en.item() * batch_samples.size(dim=0)
                    val_gen_loss += loss_gen.item() * batch_samples.size(dim=0)
                    val_dis_loss += loss_dis.item() * batch_samples.size(dim=0)

            val_en_loss = val_en_loss / test_len
            val_gen_loss = val_gen_loss / test_len
            val_dis_loss = val_dis_loss / test_len

            if (epoch + 1):
                print('Epoch %3d/%3d \n'
                      'val_en_loss %5.5f, val_gen_loss %5.5f, val_dis_loss %5.5f \n' % \
                      (epoch + 1, epochs,
                       val_en_loss, val_gen_loss, val_dis_loss))

            history['val_loss'].append(val_en_loss + val_gen_loss + val_dis_loss)

        return history
