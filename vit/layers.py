"""
Realization of the layers needed in ViT
MHSA, LayerNorm, MLP, Pos-Encoding
"""

import torch
import torch.nn as nn

from einops import rearrange

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
        # Step 1. Query, key, value calculation
        # ----------------------------------------
        # [b_size, n_seq, in_dim] -> [b_size, n_seq, 3 * qkv_dim * n_heads]
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv)
        # ----------------------------------------

        # Step 2. Attention score
        # ----------------------------------------
        # [b_size, n_seq, n_seq, n_heads]
        score = torch.einsum('bhnq, bhtq -> bhnt', q, k) * self.qkv_dim ** (-0.5)
        attn_score = torch.softmax(score, dim=-1)
        # ----------------------------------------

        # Step 3. Weighed value output
        # ----------------------------------------
        # [b_size, n_seq, qkv_dim * n_heads]
        attn_value = torch.einsum('bhgn, bhnq -> bhgq', attn_score, v)
        attn_value = rearrange(attn_value, 'b h n d -> b n (h d)')

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


# ViT
# --------------------------------------------------
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
# --------------------------------------------------