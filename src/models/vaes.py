"""
- Encoder
- Decoder
- Prior Dist
"""
import torch
import torch.nn as nn
from torch.distributions import Distribution, Normal


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        # Infer z_dim
        self.z_dim = self.encoder.z_dim

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

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.tensor:
        """Generate samples from prior dist (gradient calcs disabled)

        Parameters
        ----------
        num_samples : int
        device : torch.device

        Returns
        -------
        torch.tensor
            [B, D], Generated samples
        """
        self.eval()

        # Assume prior ~ N(0,1)
        dim = (num_samples, self.z_dim)
        z = torch.normal(mean=0.0, std=1.0, size=dim, device=device)

        loc_img = self.decoder.decode(z)

        self.train()

        return loc_img

    def reconstruct(self):
        pass
