import torch


class Generator(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, 20)  # we get input_size noise as input
        self.fc2 = torch.nn.Linear(20, 130)
        self.fc3 = torch.nn.Linear(130, output_size)
        self.bn1 = torch.nn.BatchNorm1d(130)
        self.dropout = torch.nn.Dropout(p=0.25)

    def forward(self, x):
        assert x.size(dim=1) == self.input_size, 'Dimensional of input noise should equal initialized'

        # x - input random noise with input_size dimensional
        x = torch.nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = torch.nn.ReLU()(self.bn1(self.fc2(x)))
        x = self.dropout(x)
        out = torch.nn.Tanh()(self.fc3(x))

        return out


class Discriminator(torch.nn.Module):

    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, 20)  # we get input_size noise as input
        self.fc2 = torch.nn.Linear(20, 15)
        self.fc3 = torch.nn.Linear(15, 1)  # probability of true sample as output
        self.bn1 = torch.nn.BatchNorm1d(15)
        self.dropout = torch.nn.Dropout(p=0.25)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        assert x.size(dim=1) == self.input_size, 'Dimensional of input noise should equal initialized'

        # x - input sample from true or generator distribution
        x = torch.nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = torch.nn.ReLU()(self.bn1(self.fc2(x)))
        out = torch.nn.Sigmoid()(self.fc3(x))

        return out


class GAN(torch.nn.Module):

    def __init__(self, input_size, output_size, epochs):
        super(GAN, self).__init__()
        self.input_size = input_size
        self.generator = Generator(input_size=input_size,
                                   output_size=output_size)
        self.discriminator = Discriminator(input_size=output_size)
        self.bce = torch.nn.BCELoss()

        self.g_optimizer = torch.optim.Adam(params=self.generator.parameters(), lr=3e-4)
        self.d_optimizer = torch.optim.Adam(params=self.discriminator.parameters(), lr=3e-4)
        self.epochs = epochs

    def __call__(self, X):
        # input noise
        out = self.generator(X)

        return out

    def discriminator_loss(self, pred_true, pred_fake):
        targets_true = torch.full(size=(pred_true.size(dim=0), 1), fill_value=1, dtype=torch.float32)
        targets_fake = torch.full(size=(pred_true.size(dim=0), 1), fill_value=0, dtype=torch.float32)

        bce_true = self.bce(pred_true, targets_true)
        bce_fake = self.bce(pred_fake, targets_fake)

        bce = (bce_fake + bce_true) / 2

        return bce

    def generator_loss(self, pred_fake):
        targets_fake = torch.full(size=(pred_fake.size(dim=0), 1), fill_value=1, dtype=torch.float32)
        bce_generator = self.bce(pred_fake, targets_fake)

        return bce_generator

    def fit(self, dataloader):
        G_losses = []
        D_losses = []
        img_fake = []

        for epoch in range(self.epochs):
            for i, (data, labels) in enumerate(dataloader):
                #########
                # Discriminator train stage
                # minimize -1 * {log(D(x)) + log(1 - D(G(z)))}
                # it equivalent to minimize sum of two binary cross entropy
                # first : - {y_true * log(y = 1 | x) + (1 - y_true) * (1 - log(y = 1 | x))} = - { 1 * log(y = 1 | x) }
                # second: - {y_fake * log(y = 1 | x) + (1 - y_fake) * (1 - log(y = 1 | x))} =
                # = - { (1 - y_fake) * (1 - log(y = 1 | x)) }
                # y_true = 1, y_fake = 0
                #########

                self.discriminator.zero_grad()

                b_size = data.size(0)
                Z_samples = torch.randn(b_size, self.input_size)
                fake_data = self.generator(Z_samples)

                pred_true = self.discriminator(data)
                pred_fake = self.discriminator(fake_data)

                loss_D = self.discriminator_loss(pred_true, pred_fake)
                loss_D.backward()

                self.d_optimizer.step()

                #########
                # Generator train stage
                # minimize log(1 - D(G(z))
                # it equivalent to maximize binary cross entropy
                # - {y_fake * log(y = 1 | x) + (1 - y_fake) * (1 - log(y = 1 | x))} =
                # = - { (1 - y_fake) * (1 - log(y = 1 | x)) }
                # y_fake = 0
                #########

                self.generator.zero_grad()

                Z_samples = torch.randn(b_size, self.input_size)
                fake_data = self.generator(Z_samples)

                pred_fake = self.discriminator(fake_data)

                loss_G = self.generator_loss(pred_fake)
                loss_G.backward()

                self.g_optimizer.step()

                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                          % (epoch, self.epochs, i, len(dataloader),
                             loss_D.item(), loss_G.item()))

                ###################################

                G_losses.append(loss_G.item())
                D_losses.append(loss_D.item())

                ##################################

            if epoch % 2 == 0:
                with torch.no_grad():
                    size = 3
                    noise = torch.randn(size, self.input_size)
                    fake = self.generator(noise)
                img_fake.append(fake[0])

        return img_fake
