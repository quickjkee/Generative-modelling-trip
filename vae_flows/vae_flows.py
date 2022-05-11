import torch
import time
from torch import nn


class VaeFlow(nn.Module):
    def __init__(self, encoder, decoder, flow, hidden_dim, device):
        """
        Realization of variation autoencoder with normalizing flows, can be presented as

        real object ==> encoder ==> sample ==> norm. flow ==> decoder ==> reconstructed object

        :param encoder: Encoder pytorch model
        :param decoder: Decoder pytorch model
        :param flow: Flow pytorch model
        :param hidden_dim: Dimension of latent vector
        :param device: Current working device
        """
        super(VaeFlow, self).__init__()

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.flow = flow.to(device)

        self.hidden_dim = hidden_dim
        self.device = device

        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, x):
        """
        Forward pass through VaeFlow model
        :param x: (Tensor) [B x C x W x H]
        :return: (Tensor) [B x C x W x H]
        """
        # Samples from latent encoder distribution
        mu, log_sigma = self.encoder(x)
        latent_samples = self.encoder.sample(mu, log_sigma)

        # Flow through latent samples
        flow_samples, _ = self.flow.flow(latent_samples)

        # Decoded flow samples
        decoded_samples = self.decoder(flow_samples)

        return decoded_samples

    def sample(self, num):
        """
        Generating objects from standard normal distribution samples
        :param num: (Integer) Number of samples
        :return: (Tensor) [num x C x W x H]
        """
        noise = torch.randn(size=(num, self.hidden_dim)).to(self.device)

        flow_noise, _ = self.flow.flow(noise)
        decoded_noise = self.decoder(flow_noise)

        return decoded_noise

    def elbo(self, x):
        """
        Calculating elbo:

        L = |E_p |E_q { log p(x|f(z)) + log p(f(z)) - log q(z|x) + sum log det df/dz } =
          = |E_p |E_q { ObjectPosterior + Prior - LatentPosterior + FlowDet  }
        z - samples from q(z|x), x - real object

        :param x: (Tensor) [B x C x W x H]
        :return: (Float) Calculated elbo
        """

        # Forward pass through VaeFlow
        mu, log_sigma = self.encoder(x)
        z = self.encoder.sample(mu, log_sigma)
        f_z, flow_det = self.flow.flow(z)
        x_decoded = self.decoder(f_z)

        # Calculating densities of distributions
        object_posterior = self.object_posterior(x, x_decoded)
        prior = self.prior(f_z)
        latent_posterior = self.latent_posterior(z, mu, log_sigma)

        recon_loss = -object_posterior
        kld = latent_posterior - prior - flow_det

        # ELBO loss
        elbo_loss = 0.1 * recon_loss + 150 * kld

        return elbo_loss.mean(dim=0), recon_loss.mean(dim=0), kld.mean(dim=0)

    def object_posterior(self, x, x_decoded):
        """
        Calculating log-density of object posterior distribution as

        log p(x | f(z)) = log N(x | decoder(f(z)), const) =
        = - || x - decoder(f(z)) || ^ 2

        :param x: (Tensor) [B x C x W x H] Real object
        :param x_decoded: (Tensor) [B x C x W x H] Reconstructed object
        :return: (Float)
        """
        object_posterior = -self.mse(x, x_decoded)

        return object_posterior

    def prior(self, f_z):
        """
        Calculating log-density of prior object distribution as

        log p(f(z)) = - || f(z) || ^ 2

        :param f_z: (Tensor) [B x hidden_dim] Latent samples passed through flow
        :return: (Float)
        """
        prior = (-0.5 * torch.pow(f_z, 2)).sum(dim=1)

        return prior

    def latent_posterior(self, z, mu, log_sigma):
        """
        Calculating log-density of latent posterior distribution as

        log q(z|x) = log 1 / (det se) - 1/2 || (z - mu) / se || ^ 2

        :param z: (Tensor) [B x hidden_dim] Sample from q(z|x)
        :param mu: (Tensor) [B x hidden_dim] Expectation of q(z|x)
        :param log_sigma: (Tensor) [B x hidden_dim] Log of var of q(z|x)
        :return: (Float)
        """
        latent_posterior = (-0.5 * (log_sigma + torch.pow(z - mu, 2) / torch.exp(log_sigma))).sum(dim=1)

        return latent_posterior

    def fit(self, trainloader, testloader, epochs):
        """
        Optimizing VaeFlows model via maximizing ELBO
        :param trainloader: (Dataloader) Train dataloader
        :param testloader: (Dataloader) Test dataloader
        :param epochs: (Int) Number of epochs
        :return: (dict) History of losses
        """
        flow_params = list(self.flow.parameters())
        encoder_params = list(self.encoder.parameters())
        decoder_params = list(self.decoder.parameters())

        params = encoder_params + decoder_params + flow_params
        optimizer = torch.optim.Adam(params=params, lr=3e-4)

        print('opt=%s(lr=%f), epochs=%d, device=%s\n' % \
              (type(optimizer).__name__,
               optimizer.param_groups[0]['lr'], epochs, self.device))

        history = {}
        history['loss'] = []
        history['val_loss'] = []

        for epoch in range(epochs):

            start_time = time.time()
            train_loss = 0.0

            ############ TRAINING PART ############

            for i, batch_samples in enumerate(trainloader):
                batch_samples = batch_samples[0].to(self.device)

                optimizer.zero_grad()

                elbo_loss, recon_loss, kld = self.elbo(batch_samples)
                elbo_loss.backward()

                optimizer.step()

                end_time = time.time()
                work_time = end_time - start_time

                print('Epoch/batch %3d/%3d \n'
                      'elbo loss %5.2f, recon_loss %5.2f, kld %5.2f \n'
                      'batch time %5.2f sec' % \
                      (epoch + 1, i,
                       elbo_loss.item(),
                       recon_loss.item(),
                       kld.item(),
                       work_time))

            ############ VALIDATION PART ############
            val_loss = 0.0
            test_len = len(testloader.dataset)

            with torch.no_grad():
                for i, batch_samples in enumerate(testloader):
                    batch_samples = batch_samples[0].to(self.device)

                    elbo_loss_val, recon_val_loss, kld_val_loss = self.elbo(batch_samples)

                    val_loss += elbo_loss_val.item() * batch_samples.size(dim=0)

            val_loss = val_loss / test_len

            if (epoch + 1):
                print('Epoch %3d/%3d \n'
                      'val elbo loss %5.5f\n' % \
                      (epoch + 1, epochs,
                       val_loss))

            history['val_loss'].append(val_loss)

        return history
