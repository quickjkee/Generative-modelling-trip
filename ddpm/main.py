from models.unet import UNet
import torch

unet = UNet(in_channels=3,
            n_blocks=2,
            n_channels=128,
            ch_mults=[1, 2, 2, 2])

x = torch.randn(32, 3, 32, 32)
t = torch.randn(32)

pytorch_total_params = sum(p.numel() for p in unet.parameters())

print(pytorch_total_params)
print(unet(x, t).size())