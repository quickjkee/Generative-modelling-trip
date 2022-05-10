import torch
from torch import nn

from planar_transform import PlanarTrans


class PlanarFlow(nn.Module):
    def __init__(self, len_f, h_dim, activation, der_activation):
        """
        Realization of planar flow
        :param len_f: (Integer) size of
        :param h_dim: (Integer) dimension of latent space
        :param activation: (Function) activation function
        :param der_activation (Function) derivative of activation function
        """
        super(PlanarFlow, self).__init__()

        self.len_f = len_f

        # Initializing flow (List of base transformation)
        self.transforms = [PlanarTrans(h_dim, activation, der_activation) for _ in range(self.len_f)]
        self.model = nn.Sequential(*self.transforms)

    def flow(self, z):
        """
        Residual flow and its determinant

        ### z_K = f_K o ... o f_2 o f_1 (z) - flow
        ### log det f = sum_{k=1}^K log base_der (z_{k-1}) - log of determinant

        :param z: (Tensor) [B x h_dim] hidden vector
        :return: Tuple(Tensor, Float) transformation through flow and its determinant
        """
        log_det = 0

        for trans in self.transforms:
            z, det_f = trans.base_f(z), trans.base_det(z)
            log_det += torch.log(det_f + 1e-4).view(-1)

        return z, log_det
