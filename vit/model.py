"""
Realization of the ViT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self,
                 model,
                 n_classes=10):
        """
        @param model: ViT transformer
        """
        super(Model, self).__init__()

        self.model = model
        self.n_classes = n_classes

    def cross_entropy_loss(self, pred_probs, labels):
        """
        Calculation of cross entropy according to

        H(p, q) = -E_p(x) log q(x)

        @param pred_probs: (Tensor), [b_size, n_classes]
        @param labels: (Tensor), [b_size, n_classes], labels in one hot encoding form
        @return: (Float)
        """
        prob_sum = torch.sum(labels * torch.log(pred_probs + 1e-6), dim=1)
        prob_mean_batch = -torch.mean(prob_sum, dim=0)

        return prob_mean_batch

    def accuracy_metric(self, pred_probs, labels):
        pred_labels = torch.argmax(pred_probs, dim=1)
        acc_pred = (pred_labels == labels).sum() / len(labels)

        return acc_pred

    def fit(self, n_epochs, lr, trainloader, testloader):
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f'Number of trainable parameters are {n_params}')

        opt = torch.optim.Adam(lr=lr, params=self.model.parameters())

        print('Training has been started')
        print('=======================================')

        # Training phase
        # -------------------------------
        for step in range(n_epochs):
            batch, labels = next(iter(trainloader))
            pred_probs = self.model(batch)

            one_hot_labels = F.one_hot(labels, num_classes=self.n_classes)
            loss = self.cross_entropy_loss(F.softmax(pred_probs, dim=1), one_hot_labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                print(f'Accuracy for batch {self.accuracy_metric(pred_probs, labels)}, {loss}')

        print('Training has been ended')
        print('=======================================')
