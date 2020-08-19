import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


class MNISTDecoder(nn.Module):
    def __init__(self, z_dim=20, hidden_dim=400):
        super().__init__()

        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 784)

    def decode(self, z: torch.tensor) -> torch.tensor:
        """Decode latent points into data points

        Parameters
        ----------
        z : torch.tensor
            [B, Z]

        Returns
        -------
        torch.tensor
            [B, D]
        """
        hidden = F.relu(self.fc1(z))

        # Output is Bernoulli
        loc_img = F.sigmoid(self.fc2(hidden))

        return loc_img

    def forward(self, x: torch.tensor, z: torch.tensor) -> torch.tensor:
        """log p(x|z), reconstruction term

        Parameters
        ----------
        x : torch.tensor
            [B, C, H, W], Data points
        z : torch.tensor
            [B, Z], Latent points

        Returns
        -------
        torch.tensor
            [B,], log p(x|z)
        """
        x = x.reshape(-1, 784)

        loc_img = self.decode(z)
        dist = Bernoulli(probs=loc_img)

        return dist.log_prob(x).sum(-1)
