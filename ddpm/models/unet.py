import torch.nn as nn
import torch

from models.layers import UpSample, DownSample, UnetBlock, EmbLayer, Swish


class UNet(nn.Module):
    def __init__(self, in_channels, n_channels, ch_mults):
        """
        :param in_channels: (Int), number of channels in input image
        :param n_channels: (Int), number of channels in UNet layers
        :param ch_mults: (List), multiplication factor for n_channels
        """
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.n_channels = n_channels

        self.ch_mults = ch_mults

        self.n_resol = len(ch_mults)
        self.n_embed = n_channels * 4

        self.attn = [False] * self.n_resol
        self.attn[1] = True

        self.sampling = {
            'middle': UpSample,
            'down': DownSample,
            'up': UpSample
        }

        self.in_conv = nn.Conv2d(in_channels, n_channels, 3, 1, 1)

        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.out_conv = nn.Conv2d(n_channels, in_channels, 3, 1, 1)

        self.down_blocks = self._make_blocks('down')
        self.middle_block = self._make_blocks('middle')
        self.up_blocks = self._make_blocks('up')

        self.emb = EmbLayer(n_embed=self.n_embed)

    def _make_blocks(self, type):
        blocks = nn.ModuleList()

        """
        Below some hints with dimensions, 
        if you think about it, they will become clear
        """

        out_channels = [mult * self.n_channels for mult in self.ch_mults]
        in_channels = out_channels[:-1]
        in_channels.insert(0, in_channels[0])

        if type == 'middle':
            in_channels = [out_channels[-1]]
            out_channels = [out_channels[-1] * 2]
        elif type == 'up':
            in_channels.reverse()
            out_channels.reverse()

            out_channels, in_channels = in_channels, out_channels

            in_channels = [channel * 2 for channel in in_channels]

        n_resol = 1 if type == 'middle' else self.n_resol
        is_attn = [True] * n_resol if type == 'middle' else self.attn
        sampling = self.sampling[type]

        for i in range(n_resol):
            out = out_channels[i]
            in_ = in_channels[i]

            sample = None if (sampling is None or (i == n_resol - 1 and type == 'up')) else sampling

            blocks.append(UnetBlock(in_,
                                    out,
                                    self.n_embed,
                                    sample=sample,
                                    is_attn=is_attn[i]))

        return blocks

    def _down_forward(self, x, t_emb):
        """
        :param x: (Tensor), [b_size x C x W x H]
        :param t_emb: (Tensor), [b_size x n_embed]
        :return: (List(Tensor))
        """
        outs = []

        for layer in self.down_blocks:
            x, x2 = layer(x, t_emb)
            outs.append(x2)

        return outs, x

    def _up_forward(self, x, down_outs, t_emb):
        """
        :param x: (Tensor), [b_size x C x W x H]
        :param down_outs (List(Tensor))
        :param t_emb: (Tensor), [b_size x n_embed]
        :return: (Tensor), [b_size x C x W x H]
        """
        for i, layer in enumerate(self.up_blocks):
            x = torch.cat((down_outs[-i - 1], x), dim=1)
            x, _ = layer(x, t_emb)

        return x

    def forward(self, x, t):
        """
        :param x: (Tensor), [b_size x C_in x W x H]
        :param t: (Tensor), [b_size]
        :return: (Tensor), [b_size x C_in x W x H]
        """
        # [b_size] -> [b_size x n_embed]
        t_emb = self.emb(t)

        # [b_size x C_in x W x H] -> [b_size x C x W x H]
        x_proj = self.in_conv(x)

        down_outs, x = self._down_forward(x_proj, t_emb)
        middle_out, _ = self.middle_block[0](x, t_emb)
        up_out = self._up_forward(middle_out, down_outs, t_emb)

        out = self.out_conv(self.act(self.norm(up_out)))

        return out
