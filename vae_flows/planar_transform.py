import torch
from torch import nn


class PlanarTrans(nn.Module):
    def __init__(self, h_dim, activation, der_activation):
        """
        Realization of planar transform
        :param h_dim: (Integer) dimension of latent space
        :param activation: (Function) activation function
        :param der_activation (Function) derivative of activation function
        """
        super(PlanarTrans, self).__init__()

        self.h_dim = h_dim

        # Initialization of parameters of flow
        self.w = nn.Parameter(torch.randn(self.h_dim).normal_(0, 0.1))
        self.b = nn.Parameter(torch.randn(1).normal_(0, 0.1))
        self.u = nn.Parameter(torch.randn(self.h_dim).normal_(0, 0.1))

        self.activation = activation
        self.der_activation = der_activation

    def base_f(self, z):
        """
        Base function of residual flow

        ### f(z) = z + uh(w^T * z + b)
        ### h - activation function; w,b,u - weights

        :param z: (Tensor)[B x h_dim] hidden vector
        :return: (Tensor) [B x h_dim] transformation throught function
        """
        assert z.size(1) == self.h_dim, 'Input size should correspond to hidden dim'

        # checking sufficient condition for reversibility
        if torch.matmul(self.w, self.u) < -1:
            self.u_modification()

        f = z + self.u * self.activation(torch.matmul(z, self.w).view(-1, 1) + self.b)

        return f

    def u_modification(self):
        """
        Modification of parameter u to satisfy condition for reversibility
        :return: None
        """
        w_u = torch.matmul(self.w, self.u)
        m_w_u = -1 + torch.log(1 + torch.exp(w_u))
        w_norm = self.w / torch.norm(self.w, p=2) ** 2

        self.u.data = self.u + (m_w_u - w_u) * w_norm

    def base_det(self, z):
        """
        Determinant of base function

        ### det f = | 1 + u^T * h'(w^T * z + b)w |

        :param z: (Tensor) [B x h_dim] hidden vector
        :return: (Tensor) [B x 1] determinant of base function
        """
        assert z.size(1) == self.h_dim, 'Input size should correspond to hidden dim'

        if torch.matmul(self.w, self.u) < -1:
            self.u_modification()

        affine_trans = torch.matmul(z, self.w).view(-1, 1) + self.b
        phi = (1 - torch.tanh(affine_trans) ** 2) * self.w
        df_dz = torch.abs(1 + torch.matmul(phi, self.u))

        return df_dz
