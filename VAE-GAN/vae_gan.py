import torch
import time

from torch import nn


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

        self.bce = nn.BCELoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')

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

        return object_sample

    def prior_loss(self, x):
        """
        Calculating KL-divergence between q(z|x) and N(0, I)
        :param x: (Tensor) [B x C x W x H]
        :return: (Float) Value of KL divergence
        """
        mu, log_sigma = self.encoder(x)
        loss = (-0.5 * (1 + log_sigma - torch.exp(log_sigma) - mu ** 2).sum(dim=1)).mean(dim=0)

        return loss

    def gan_loss(self, x, key):
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

        noise = torch.randn(size=(batch_size, self.hidden_dim)).to(self.device)

        if key:
            real_loss = self.bce(self.discriminator(x), ones)
        else:
            real_loss = 0

        x_de = self.__call__(x)
        recon_loss = self.bce(self.discriminator(x_de), zeros)

        x_noise = self.decoder.sample(noise)
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
        D_l_x = self.discriminator.conv_out(x)

        x_de = self.__call__(x)
        D_l_x_de = self.discriminator.conv_out(x_de)

        hidden_loss = self.mse(D_l_x, D_l_x_de)

        return hidden_loss

    def encoder_loss(self, x):
        """
        Calculating encoder loss based on formula

        L_encoder = L_prior + L_l

        :param x: (Tensor) [B x C x W x H]
        :return: (Float)
        """
        l_prior = self.prior_loss(x)
        l_l = self.hidden_loss(x)

        l_encoder = l_prior + l_l

        return l_encoder

    def generator_loss(self, x):
        """
        Calculating generator (decoder) loss based on formula

        L_gen = -L_gan + L_l

        :param x: (Tensor) [B x C x W x H]
        :return: (Float)
        """
        l_gan = self.gan_loss(x, key=False)
        l_l = self.hidden_loss(x)

        l_gen = l_l - l_gan

        return l_gen

    def discriminator_loss(self, x):
        """
        Calculating discriminator loss based on formula

        L_dis = L_gan

        :param x: (Tensor) [B x C x W x H]
        :return: (Float)
        """

        l_dis = self.gan_loss(x, key=True)

        return l_dis

    def fit(self, trainloader, testloader, epochs):
        """
        Optimizing VAE/GAN model
        :param trainlaoder: (Dataloader) Train dataloader
        :param testloader: (Dataloader) Test dataloader
        :param epochs: (Int) Number of epochs
        :return: (dict) History of losses
        """

        def optim_step(optim, loss_f, batch_sample):
            """
            Another step of optimizer
            :param optim: Certain optimizer
            :param loss: (Function)
            :param batch_sample: (Tensor) [B x C x W x H]
            :return:
            """
            optim.zero_grad()

            loss = loss_f(batch_sample)
            loss.backward()

            optim.step()

            elem_loss = loss * batch_sample.size(0)

            return elem_loss

        params = {
            'encoder': list(self.encoder.parameters()),
            'generator': list(self.decoder.parameters()),
            'discriminator': list(self.discriminator.parameters())
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

        for epoch in range(epochs):
            start_time = time.time()

            ############ TRAINING PART ############

            train_en_loss = 0.0
            train_gen_loss = 0.0
            train_dis_loss = 0.0

            for i, batch_samples in enumerate(trainloader):
                batch_samples = batch_samples[0].to(self.device)

                ##############
                # ENCODER FIT
                ##############

                train_en_loss += optim_step(en_optim, self.encoder_loss, batch_samples)

                ##############
                # GENERATOR FIT
                ##############

                train_gen_loss += optim_step(gen_optim, self.generator_loss, batch_samples)

                ##############
                # DISCRIMINATOR FIT
                ##############

                train_dis_loss += optim_step(dis_optim, self.discriminator_loss, batch_samples)

            train_en_loss = train_en_loss / (len(trainloader.dataset))
            train_gen_loss = train_gen_loss / (len(trainloader.dataset))
            train_dis_loss = train_dis_loss / (len(trainloader.dataset))

            ############ VALIDATION PART ############

            val_en_loss = 0.0
            val_gen_loss = 0.0
            val_dis_loss = 0.0

            with torch.no_grad():
                for i, batch_samples in enumerate(testloader):
                    batch_samples = batch_samples[0].to(self.device)

                    en_loss = self.encoder_loss(batch_samples)
                    gen_loss = self.generator_loss(batch_samples)
                    dis_loss = self.discriminator_loss(batch_samples)

                    val_en_loss = en_loss.item() * batch_samples.size(0)
                    val_gen_loss = gen_loss.item() * batch_samples.size(0)
                    val_dis_loss = dis_loss.item() * batch_samples.size(0)

            val_en_loss = val_en_loss / (len(testloader.dataset))
            val_gen_loss = val_gen_loss / (len(testloader.dataset))
            val_dis_loss = val_dis_loss / (len(testloader.dataset))

            end_time = time.time()
            work_time = end_time - start_time

            if (epoch + 1):
                print('Epoch %3d/%3d \n'
                       'train_en_loss %5.5f, train_gen_loss %5.5f, train_dis_loss %5.5f \n'
                       'val_en_loss %5.5f, val_gen_loss %5.5f, val_dis_loss %5.5f \n'
                       'epoch time %5.2f sec' % \
                      (epoch + 1, epochs,
                       train_en_loss, train_gen_loss, train_dis_loss,
                       val_en_loss, val_gen_loss, val_dis_loss,
                       work_time))

            history['loss'].append(train_en_loss + train_gen_loss + train_dis_loss)
            history['val_loss'].append(val_en_loss + val_gen_loss + val_dis_loss)

        return history
