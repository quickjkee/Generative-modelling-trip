from torch.nn.modules.activation import Tanh
import torch
import time

from torch import nn


class Encoder(nn.Module):
    def __init__(self, data_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(1024, 2 * hidden_dim),
            nn.ReLU(),
        )

        self.mu = None  # Expectation of Q(Z|X, Phi) distribution
        self.log_sigma = None  # Dispersion of Q(Z|X, Phi) distribution

    def forward(self, x):
        assert x.size(dim=1) == self.data_dim, 'Wrong input size'

        out = self.model(x)
        self.mu = out[:, :self.hidden_dim]
        self.log_sigma = out[:, self.hidden_dim:]

        return self.mu, self.log_sigma

    def sample(self, x):
        # Latent representation of input object
        assert self.mu is not None, 'Latent distribution are not prepared'
        assert x.size(dim=1) == self.data_dim, 'Wrong input size'

        eps = torch.randn(x.size(dim=0), self.hidden_dim).to(device)  # Samples from standard normal distribution
        mu, log_sigma = self.forward(x)

        latent_samples = torch.exp(0.5 * log_sigma) * eps + mu

        return latent_samples


class Decoder(nn.Module):
    def __init__(self, data_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(1024, data_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        assert x.size(dim=1) == self.hidden_dim, 'Wrong input size'

        out = self.model(x)

        return out

    def sample(self, noise):
        # Sampling an object from input noise, i.e. from N(0, I)
        assert noise.size(dim=1) == self.hidden_dim

        objects_samples = self.model(noise)

        return objects_samples


class VAE(nn.Module):
    def __init__(self, data_dim, hidden_dim, device):
        super(VAE, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.encoder = Encoder(data_dim, hidden_dim).to(device)
        self.decoder = Decoder(data_dim, hidden_dim).to(device)

        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCELoss(reduction='sum')

    def __call__(self, x):
        assert x.size(dim=1) == self.data_dim, 'Wrong input size'

        # Getting a similar object as input

        latent_sample = self.encoder.sample(x)
        object_sample = self.decoder.sample(latent_sample)

        return object_sample

    def kl_divergence(self, x):
        # Calculation KL divergence between KL(Q(z|x) || P(z))
        mu, log_sigma = self.encoder(x)
        loss = -0.5 * torch.sum(1 + log_sigma - torch.exp(log_sigma) - mu ** 2)

        return loss

    def expectation(self, x):
        # Calculation of |E_eps~N(0,I) {log p(x|eps*sigma + mu)}
        latent_sample = self.encoder.sample(x)
        decoder_output = self.decoder(latent_sample)

        loss = self.bce(decoder_output, x)

        return loss

    def loss_vae(self, x):
        # ELBO loss representation
        batch_size = x.size(dim=0)

        vae_loss = 1 / (batch_size) * (self.kl_divergence(x) + self.expectation(x))

        return vae_loss

    def fit(self, trainloader, testloader, epochs):
        encoder_params = list(self.encoder.parameters())
        decoder_params = list(self.decoder.parameters())
        params = encoder_params + decoder_params
        optimizer = torch.optim.Adam(params=params, lr=0.0001)

        print('opt=%s(lr=%f), epochs=%d, device=%s\n' % \
              (type(optimizer).__name__,
               optimizer.param_groups[0]['lr'], epochs, self.device))

        history = {}
        history['loss'] = []
        history['val_loss'] = []

        for epoch in range(epochs):

            start_time = time.time()
            train_loss = 0.0

            for i, batch_samples in enumerate(trainloader):
                batch_samples = batch_samples[0].view(-1, self.data_dim).to(self.device)

                optimizer.zero_grad()

                loss = self.loss_vae(batch_samples)
                loss.backward()

                optimizer.step()

                train_loss += loss.item() * batch_samples.size(0)

            train_loss = train_loss / len(trainloader.dataset)

            ################### VALIDATION PART ###########################

            val_loss = 0.0

            with torch.no_grad():
                for i, batch_samples in enumerate(testloader):
                    batch_samples = batch_samples[0].view(-1, self.data_dim).to(self.device)

                    loss = self.loss_vae(batch_samples)
                    val_loss += loss.item() * batch_samples.size(0)

            val_loss = val_loss / len(testloader.dataset)

            end_time = time.time()
            work_time = end_time - start_time

            if (epoch + 1):
                print('Epoch %3d/%3d, train_loss %5.5f, val_loss %5.5f, epoch time %5.2f sec' % \
                      (epoch + 1, epochs, train_loss, val_loss, work_time))

            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)

        return history
