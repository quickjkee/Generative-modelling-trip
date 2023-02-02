"""
Realization of the layers needed in ViT
MHSA, LayerNorm, MLP, Pos-Encoding
"""

import torch
import torch.nn as nn


# Positional encoding layer
# --------------------------------------------------
class PosEncoding(nn.Module):
    """
    Positional encoding according to "Attention is all you need"

    H = {h_1 + PE_1, ..., h_N + PE_N}, PE_i R^{in_dim}
    PE_i = { sin(k / 1000 ** (2 / in_dim)), cos(k / 1000 ** (2 / in_dim)), ..., cos(k / 1000 ** (2N / in_dim))}
    """

    def __init__(self, n_seq, in_dim):
        """
        @param n_seq: (Int), length of the sequence
        @param in_dim: (Int), each element dimension
        """
        super(PosEncoding, self).__init__()

        self.n_seq = n_seq
        self.in_dim = in_dim

        self._init_encoding()

    def _init_encoding(self):
        """
        Initialization of the positional encoding vectors
        @return: None
        """
        positions = torch.arange(start=0, end=self.n_seq, step=1)
        frequencies = 10000. ** (torch.arange(start=0, end=self.in_dim, step=2) / self.in_dim)

        encoding = torch.zeros(self.n_seq, self.in_dim)
        encoding[:, 0::2] = torch.sin(positions[:, None] / frequencies)
        encoding[:, 1::2] = torch.cos(positions[:, None] / frequencies)

        self.encoding = encoding

    def forward(self, x):
        """
        @param x: (Tensor), [b_size, n_seq, in_dim]
        @return: (Tensor), [b_size, n_seq, in_dim]
        """
        pos_emb = x + self.encoding
        return pos_emb
# --------------------------------------------------


# Multi Layer Perceptron
# --------------------------------------------------
class MLP(nn.Module):
    """
    Single MLP layer
    """

    def __init__(self, in_dim, out_dim):
        """
        @param in_dim: (Int), input dimension
        @param out_dim: (Int), output dimension
        """
        super(MLP, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.model = nn.Sequential(
            nn.Linear(self.in_dim,
                      self.out_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(self.out_dim,
                      self.in_dim),
            nn.Dropout(0.25),

        )

    def forward(self, x):
        """
        @param x: (Tensor), [b_size, n_seq, in_dim]
        @return: (Tensor), [b_size, n_seq, out_dim]
        """
        out = self.model(x)

        return out
# --------------------------------------------------


# Multi Head Self Attention
# --------------------------------------------------
class MHSA(nn.Module):
    def __init__(self, in_dim, n_heads):
        super(MHSA, self).__init__()

        self.in_dim = in_dim

        self.qkv_dim = in_dim
        self.n_heads = n_heads

        self.qkv = nn.Linear(in_dim, 3 * self.qkv_dim * self.n_heads)
        self.w_0 = nn.Linear(self.n_heads * self.qkv_dim, self.in_dim)

    def forward(self, x):
        """
        @param x: (Tensor), [b_size, n_seq, in_dim]
        @return: (Tensor), [b_size, n_seq, in_dim]
        """
        b_size, n_seq, in_dim = x.size()

        # Step 1. Query, key, value calculation
        # ----------------------------------------

        # [b_size, n_seq, in_dim] -> [b_size, n_seq, 3 * qkv_dim * n_heads]
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv.view((3, b_size, n_seq, self.qkv_dim, self.n_heads)), [1, 1, 1], dim=0)
        # ----------------------------------------

        # Step 2. Attention score
        # ----------------------------------------

        # [b_size, n_seq, n_seq, n_heads]
        score = torch.einsum('bnqh, btqh -> bnth', q.squeeze(), k.squeeze()) * self.qkv_dim ** (-0.5)
        attn_score = torch.softmax(score, dim=1)
        # ----------------------------------------

        # Step 3. Weighed value output
        # ----------------------------------------

        # [b_size, n_seq, qkv_dim * n_heads]
        attn_value = torch.einsum('bnnh, bnqh -> bnqh', attn_score, v.squeeze())
        attn_value = attn_value.view(b_size, n_seq, self.qkv_dim * self.n_heads)

        out = self.w_0(attn_value)
        # ----------------------------------------

        return out
# --------------------------------------------------


# Transformer block
# --------------------------------------------------
class TransBlock(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads):
        super(TransBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)

        self.mhsa = MHSA(in_dim, n_heads)
        self.mlp = MLP(in_dim, out_dim)

    def forward(self, x):
        """
        @param x: (Tensor), [b_size, n_seq, in_dim]
        @return: (Tensor), [b_size, n_seq, in_dim]
        """

        # Step 1. MHSA and residual connection
        mhsa_out = self.mhsa(self.norm1(x))
        res_out = mhsa_out + x

        # Step 2. MLP and residual connection
        mlp_out = self.mlp(self.norm2(res_out))
        res_out = mlp_out + res_out

        return res_out
# --------------------------------------------------
