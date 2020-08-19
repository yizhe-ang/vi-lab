"""
- Encoder
- Decoder
- Prior Dist
"""
import torch.nn as nn
import torch
from torch.distributions import Normal, Distribution


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def prior(self, z: torch.tensor) -> Distribution:
        """Assume prior is standard normal, p(z) ~ Normal(0, 1)

        Parameters
        ----------
        z : torch.tensor
            [B, D], Latent points (to infer the shapes)

        Returns
        -------
        Distribution
            Standard normal distribution of latent points
        """
        return Normal(torch.zeros_like(z), torch.ones_like(z))

    def forward(self, x: torch.tensor):
        pass

    def sample(self):
        pass

    def reconstruct(self):
        pass
