import torch
import time

from torch import nn


class Encoder(nn.Module):
    def __init__(self, hidden_dim, conv_dims, device):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.device = device

        self.conv_dims = conv_dims  # List of dims

        modules = []

        in_channels = 1
        for c_dim in conv_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=c_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(c_dim),
                    nn.LeakyReLU())
            )
            in_channels = c_dim

        modules.append(nn.AdaptiveAvgPool2d(output_size=1))
        modules.append(nn.Flatten())

        self.model = nn.Sequential(*modules)

        self.mu = nn.Linear(conv_dims[-1], hidden_dim)  # Expectation of Q(Z|X, Phi) distribution
        self.log_sigma = nn.Linear(conv_dims[-1], hidden_dim)  # Dispersion of Q(Z|X, Phi) distribution

    def forward(self, x):
        out = self.model(x)
        mu = self.mu(out)
        log_sigma = self.log_sigma(out)

        return mu, log_sigma

    def sample(self, x):
        # Latent representation of input object
        assert self.mu is not None, 'Latent distribution is not prepared'

        eps = torch.randn(x.size(dim=0), self.hidden_dim).to(self.device)  # Samples from standard normal distribution
        mu, log_sigma = self.forward(x)

        latent_samples = torch.exp(0.5 * log_sigma) * eps + mu

        return latent_samples


class Decoder(nn.Module):
    def __init__(self, hidden_dim, conv_dims, device):
        super(Decoder, self).__init__()

        self.hidden_dim = hidden_dim

        self.device = device

        self.conv_dims = conv_dims
        self.conv_dims.reverse()

        self.input_decoder = nn.Linear(hidden_dim, self.conv_dims[0] * 16)

        modules = []
        for i in range(len(self.conv_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.conv_dims[i],
                                       self.conv_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(self.conv_dims[i + 1]),
                    nn.ReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(self.conv_dims[-1],
                               self.conv_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(self.conv_dims[-1]),
            nn.ReLU(),
            nn.Conv2d(self.conv_dims[-1], out_channels=1,
                      kernel_size=3, padding=1),
            nn.Sigmoid())

        self.model = None

    def forward(self, x):
        # assert x.size(dim=1) == self.hidden_dim, 'Wrong input size'

        result = self.input_decoder(x)
        result = torch.flatten(result)
        result = result.view(-1, self.conv_dims[0], 4, 4)
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def sample(self, noise):
        # Sampling an object from input noise, i.e. from N(0, I)
        assert noise.size(dim=1) == self.hidden_dim

        objects_samples = self.forward(noise)

        return objects_samples


class VAE(nn.Module):
    def __init__(self, hidden_dim, conv_dims, device):
        super(VAE, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device

        self.encoder = Encoder(hidden_dim, conv_dims, device).to(device)
        self.decoder = Decoder(hidden_dim, conv_dims, device).to(device)

        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCELoss(reduction='sum')

    def __call__(self, x):
        # Getting a similar object as input

        latent_sample = self.encoder.sample(x)
        object_sample = self.decoder.sample(latent_sample)

        return object_sample

    def kl_divergence(self, x):
        # Calculation KL divergence between KL(Q(z|x) || P(z))
        mu, log_sigma = self.encoder(x)
        loss = (-0.5 * (1 + log_sigma - torch.exp(log_sigma) - mu ** 2).sum(dim=1)).mean(dim=0)

        return loss

    def expectation(self, x):
        # Calculation of |E_eps~N(0,I) {log p(x|eps*sigma + mu)}
        batch_size = x.size(dim=0)

        latent_sample = self.encoder.sample(x)
        decoder_output = self.decoder(latent_sample)

        loss = self.bce(decoder_output, x) / batch_size

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

            for i, batch_samples in enumerate(trainloader):
                batch_samples = batch_samples[0].to(self.device)

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
                    batch_samples = batch_samples[0].to(self.device)

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
