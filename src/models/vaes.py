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

    def forward(self, x: torch.tensor, indices=None, K=1):
        """Return components required for loss computation

        Parameters
        ----------
        x : torch.tensor
            [description]
        K : int, optional
            For IWAE, by default 1

        Returns
        -------
        [type]
            [B,],     [B,],     [B,]
            log_qz_x, log_px_z, log_pz, qz_x, pz
        """
        log_qz_x, z, qz_x = self.encoder(x, indices=indices, K=K)
        pz = self.prior(z)
        log_pz = pz.log_prob(z).sum(-1)

        log_px_z = self.decoder(x, z)

        return log_qz_x, log_px_z, log_pz, qz_x, pz

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
