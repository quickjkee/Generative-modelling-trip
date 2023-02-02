"""
Realization of the ViT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from layers import PosEncoding, TransBlock


class ViT(nn.Module):
    def __init__(self,
                 n_classes=5,
                 dim=512,
                 mlp_dim=512,
                 n_blocks=5,
                 n_heads=8,
                 img_size=32,
                 in_channels=1,
                 patch_size=4):
        """
        @param dim: Latent dimension of MHSA
        @param mlp_dim: Dimension for MLP (dim -> mlp_dim -> dim)
        @param n_blocks: Number of encoding blocks
        @param n_heads: Number of heads in MHSA
        @param patch_size: Patch size for single dimension of image (e.g, for width)
        @param in_channels: Channel dimension of input image
        @param n_classes: Number of classes to predict
        """
        super(ViT, self).__init__()

        assert img_size % patch_size == 0, 'Image size must be divisible by the patch size'

        self.patch_size = int(patch_size)
        self.n_seq = int(img_size ** 2 / patch_size ** 2)
        self.n_classes = n_classes

        # embedding
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.embed_layer = nn.Linear(self.patch_size ** 2 * in_channels, dim)
        self.pos_encod = PosEncoding(n_seq=self.n_seq + 1,  # plus one since class emb
                                     in_dim=dim)

        # transformer encoder
        self.transformer = nn.Sequential(
            *[TransBlock(dim, mlp_dim, n_heads) for _ in range(n_blocks)]
        )

        # mlp head
        self.mlp_head = nn.Linear(dim, n_classes)

        self._init_params()

    def _init_params(self, default_initialization=False):
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward propagation of the whole ViT
        @param x: (Tensor), [b_size, C, W, H], input image
        @return: (Tensor), [b_size, n_classes], probs of classes
        """

        # Step 1. Patching, embedding and pos encoding
        # ----------------------------------------

        # [b_size, n_seq, patch_size **2 * channels]
        patch_img = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                              p1=self.patch_size,
                              p2=self.patch_size)
        # [b_size, n_seq, dim]
        emb_img = self.embed_layer(patch_img)
        # [b_size, n_seq + 1, dim]
        emb_img_cls = torch.cat((self.cls_token[None, None, :].repeat(emb_img.size(0), 1, 1), emb_img),
                                dim=1)
        # [b_size, n_seq + 1, dim]
        emb_pos = self.pos_encod(emb_img_cls)
        # ----------------------------------------

        # Step 2. Transformer encoding
        # ----------------------------------------

        # [b_size, n_seq + 1, dim]
        out_trans = self.transformer(emb_pos)
        # ----------------------------------------

        # Step 3. Mlp head
        # ----------------------------------------

        # [b_size, dim]
        cls_token = out_trans[:, 0, :].squeeze()
        # [b_size, n_classes]
        prob_out = self.mlp_head(cls_token)
        # ----------------------------------------

        return prob_out

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
        n_params = sum(p.numel() for p in self.parameters())
        print(f'Number of trainable parameters are {n_params}')

        opt = torch.optim.Adam(lr=lr, params=self.parameters())

        print('Training has been started')
        print('=======================================')

        # Training phase
        # -------------------------------
        for step in range(n_epochs):
            batch, labels = next(iter(trainloader))
            pred_probs = self.forward(batch)

            one_hot_labels = F.one_hot(labels, num_classes=self.n_classes)
            loss = self.cross_entropy_loss(F.softmax(pred_probs, dim=1), one_hot_labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                print(f'Accuracy for batch {self.accuracy_metric(pred_probs, labels)}')

        print('Training has been ended')
        print('=======================================')
