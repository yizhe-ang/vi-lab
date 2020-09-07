from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal

from src.mcmc import LangevinMCMC
from src.models.nns import Encoder


class ConvEncoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()

        self.encoder = Encoder(
            128,
            z_dim,
            1,
            28,
            28
        )
        self.z_dim = z_dim

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        z_loc, z_scale = self.encoder(x)

        return z_loc, z_scale

    def forward(
        self, x: torch.Tensor, indices=None, K=1
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Distribution]]:
        """Forward pass through encoder, returning components required to compute
        loss functions

        Parameters
        ----------
        x : torch.tensor
            [B, C, H, W], Input points
        K : int
            For IWAE

        Returns
        -------
        Tuple[torch.tensor, torch.tensor, Distribution]
            [B,],       [B, Z], [B, Z]
            log q(z|x), z,      q(z|x)
        """
        z_loc, z_scale = self.encode(x)

        qz_x = Normal(z_loc, z_scale)

        if K == 1:
            z = qz_x.rsample()
        else:
            size = torch.Size([K])
            z = qz_x.rsample(size)

        log_qz_x = qz_x.log_prob(z).sum(-1)

        return log_qz_x, z, qz_x


class MNISTEncoder(nn.Module):
    def __init__(self, z_dim, hidden_dim=400):
        super().__init__()

        self.z_dim = z_dim

        self.fc1 = nn.Linear(784, hidden_dim)

        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self, x: torch.Tensor, indices=None, K=1
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Distribution]]:
        """Forward pass through encoder, returning components required to compute
        loss functions

        Parameters
        ----------
        x : torch.tensor
            [B, C, H, W], Input points
        K : int
            For IWAE

        Returns
        -------
        Tuple[torch.tensor, torch.tensor, Distribution]
            [B,],       [B, Z], [B, Z]
            log q(z|x), z,      q(z|x)
        """
        z_loc, z_scale = self.encode(x)

        qz_x = Normal(z_loc, z_scale)

        if K == 1:
            z = qz_x.rsample()
        else:
            size = torch.Size([K])
            z = qz_x.rsample(size)

        log_qz_x = qz_x.log_prob(z).sum(-1)

        return log_qz_x, z, qz_x


class MNISTLangevinEncoder(MNISTEncoder):
    def __init__(
        self,
        z_dim: int,
        dataset_size: int,
        device: torch.device,
        eps: float,
        s: int,
        t: int,
        hidden_dim=400,
    ) -> None:
        super().__init__(z_dim, hidden_dim)

        self.eps = eps
        self.s = s
        self.t = t
        # HACK Just put on cpu instead?
        self.cached_samples = torch.randn(dataset_size, z_dim).cuda()

    def forward(
        self, x: torch.Tensor, indices: torch.Tensor = None, K=1
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Distribution]]:

        z_loc, z_scale = self.encode(x)
        qz_x = Normal(z_loc, z_scale)

        # Get cached samples
        z = self.cached_samples[indices]

        # Sample using Langevin
        # FIXME Something wrong with sampler
        mcmc = LangevinMCMC(qz_x, self.eps)

        z = mcmc.simulate(z, self.s, create_graph=False)
        z = mcmc.simulate(z, self.t, create_graph=True)

        # FIXME Cache samples; do I have to create a copy on top of detaching?
        self.cached_samples[indices] = z.detach().clone()

        log_qz_x = qz_x.log_prob(z).sum(-1)

        print('Posterior term #######################')
        print(log_qz_x)

        return log_qz_x, z, qz_x


class MNISTFlowEncoder(MNISTEncoder):
    def __init__(self, z_dim, flow, hidden_dim=400):
        super().__init__(z_dim, hidden_dim)

        self.flow = flow

    def forward(
        self, x: torch.Tensor, indices=None, K=1
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Distribution]]:
        # Get initial distribution and samples
        log_qz0_x, z0, qz0_x = super().forward(x, K)

        # Flow to get final sample
        zk, log_det = self.flow(z0)

        # FIXME minus or plus? (seems to be minus)
        log_qzk_x = log_qz0_x - log_det

        return log_qzk_x, zk, None
