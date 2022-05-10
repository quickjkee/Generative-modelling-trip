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

        self.bce = nn.BCELoss(reduction='sum')
        self.mse = nn.MSELoss(reduction='sum')

        self.rate = 1e-5

    def __call__(self, x):
        """
        Only the encoder and decoder is used for the call
        Sampling from latent distribution q(z|x) and
        reconstructing with decoder
        :param x: (Tensor) [B x C x W x H]
        :return: (Tensor) Similar as input object
        """
        latent_sample = self.encoder.sample(x)
        object_sample = self.decoder.forward(latent_sample)
        probab = self.discriminator(object_sample)

        return probab

    def prior_loss(self, x):
        """
        Calculating KL-divergence between q(z|x) and N(0, I)
        :param x: (Tensor) [B x C x W x H]
        :return: (Float) Value of KL divergence
        """
        mu, log_sigma = self.encoder(x)
        loss = torch.sum(-0.5 * (1 + log_sigma - torch.exp(log_sigma) - mu ** 2))

        return loss

    def gan_loss(self, x):
        """
        Calculating gan_loss which is equal

        L_gan = binary_ce(D(x), ones) + binary_ce(D(x_de), zeros) + binary_ce(D(x_noise), zeros),

        ## x - real objects
        ## x_de = G(E(x)) - decoded samples from latent distribution q(z|x)
        ## x_noise = G(noise) - decoded samples from noise p(z) = N(0, I)
        ## binary_ce - binary cross entropy

        :param x: (Tensor) [B x C x W x H]
        :param key: (Int) Binary value, 1 - train D, 0 - train G
        :return: (Float) Value of GAN loss
        """
        batch_size = x.size(dim=0)

        ones = torch.ones(size=(batch_size, 1)).to(self.device)
        zeros = torch.zeros(size=(batch_size, 1)).to(self.device)

        noise = Variable(torch.randn(size=(batch_size, self.hidden_dim))).to(self.device)

        real_loss = self.bce(self.discriminator(x), ones)

        x_de = self.decoder(self.encoder.sample(x))
        recon_loss = self.bce(self.discriminator(x_de), zeros)

        x_noise = self.decoder(noise)
        fake_loss = self.bce(self.discriminator(x_noise), zeros)

        gan_loss = real_loss + recon_loss + fake_loss

        return gan_loss

    def hidden_loss(self, x):
        """
        Calculating reconstruction loss with following formula
        using reparameterizarion trick and M-C estimation

        -|E_q [ log{p(D_l(x)|z)} ] ~ -log{p(D_l(x)|z)}

        ## D_l(x) - output of the l-th layer of the discriminator
        ## p(D_l(x)|z) = N(D_l(x) | D_l(x_de), I) => -log{p(D_l(x)|z)} = +||D_l(x) - D_l(x_de)||^2
        ## x_de = G(E(x)) - decoded samples from latent distribution q(z|x)
        ## E(x) means sample from q(z|x)

        :param x: (Tensor) [B x C x W x H]
        :return: (Float) Value of hidden discriminator loss
        """
        d_l_x = self.discriminator.conv_out(x)

        x_de = self.decoder(self.encoder.sample(x))
        d_l_x_de = self.discriminator.conv_out(x_de)

        hidden_loss = torch.sum(0.5 * (d_l_x - d_l_x_de) ** 2)

        return hidden_loss

    def get_losses(self, x):
        """
        Calculating losses for encoder, generator, discriminator
        :param x: (Tensor) [B x C x W x H]
        :return: List(Float) Three losses
        """
        batch_size = x.size(dim=0)

        ########
        # Calculating encoder loss based on formula
        #      L_encoder = L_prior + L_l
        ########

        l_prior = self.prior_loss(x)
        l_l = self.hidden_loss(x)

        l_encoder = (l_prior + l_l) / batch_size

        ########
        # Calculating generator loss based on formula
        #      L_gan = -L_gan + L_l
        ########

        l_gan = self.gan_loss(x)

        l_gen = (self.rate * l_l - l_gan) / batch_size

        ########
        # Calculating generator loss based on formula
        #      L_dis = L_gan
        ########

        l_dis = l_gan / batch_size

        return l_encoder, l_gen, l_dis

    def fit(self, trainloader, testloader, epochs):
        """
        Optimizing VAE/GAN model
        :param trainlaoder: (Dataloader) Train dataloader
        :param testloader: (Dataloader) Test dataloader
        :param epochs: (Int) Number of epochs
        :return: (dict) History of losses
        """
        params = {
            'encoder': self.encoder.parameters(),
            'generator': self.decoder.parameters(),
            'discriminator': self.discriminator.parameters()
        }

        en_optim = torch.optim.Adam(params=params['encoder'], lr=3e-4)
        gen_optim = torch.optim.Adam(params=params['generator'], lr=3e-4)
        dis_optim = torch.optim.Adam(params=params['discriminator'], lr=3e-4)

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

                loss_en.backward(retain_graph=True, inputs=params['encoder'])
                en_optim.step()

                ##############
                # GENERATOR FIT
                ##############
                gen_optim.zero_grad()

                loss_gen.backward(retain_graph=True, inputs=params['generator'])
                gen_optim.step()

                ##############
                # DISCRIMINATOR FIT
                ##############
                dis_optim.zero_grad()

                loss_dis.backward(inputs=params['discriminator'])
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
