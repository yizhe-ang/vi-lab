import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Tuple
from torch.distributions import Normal, Distribution, Optional


class MNISTEncoder(nn.Module):
    def __init__(self, z_dim=20, hidden_dim=400):
        super().__init__()

        self.fc1 = nn.Linear(784, hidden_dim)

        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)

    def encode(self, x: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """Encode batch of points into loc and scale of latent dist

        Parameters
        ----------
        x : torch.tensor
            [B, C, H, W], Batch of points

        Returns
        -------
        Tuple[torch.tensor, torch.tensor]
            [B, Z], [B, Z]
            z_loc,  z_scale
        """
        x = x.reshape(-1, 784)

        hidden = F.relu(self.fc1(x))

        z_loc = self.fc21(hidden)
        z_scale = self.fc22(hidden).exp()

        return z_loc, z_scale

    def forward(
        self, x: torch.tensor, reparam=True
    ) -> Tuple[torch.tensor, torch.tensor, Optional[Distribution]]:
        """Forward pass through encoder, returning components required to compute
        objective

        Parameters
        ----------
        x : torch.tensor
            [B, C, H, W], Input points
        reparam : bool, optional
            Whether to return reparameterized samples, by default True

        Returns
        -------
        Tuple[torch.tensor, torch.tensor, Distribution]
            [B,],       [B, Z],  Distribution
            log q(z|x), samples, q(z|x)
        """
        z_loc, z_scale = self.encode(x)

        qz_x = Normal(z_loc, z_scale)
        samples = qz_x.rsample() if reparam else qz_x.sample()
        log_qz_x = qz_x.log_prob(samples).sum(-1)

        return log_qz_x, samples, qz_x
