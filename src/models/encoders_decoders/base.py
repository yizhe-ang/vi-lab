"""In general:

Encoders:
- Encode data point into context features,
    - For e.g. features of length latent_dim * 2

Decoders:
- Decode latent point into parameters of likelihood distribution
- Shape of output should correspond to data point,
    - For e.g. reshape to dim of image
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import Swish

__all__ = [
    "MNISTEncoder1",
    "DeepMNISTEncoder",
    "MNISTDecoder1",
    "SVHNEncoder",
    "DeepSVHNEncoder",
    "SVHNDecoder",
    "PartitionedMNISTEncoder",
    "PartitionedSVHNEncoder",
]


class MNISTEncoder2(nn.Module):
    """Simple MNIST encoder with one hidden layer, from MMVAE paper"""

    def __init__(self, latent_dim):
        super().__init__()

        self.fc1 = nn.Linear(784, 400)

        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 784)

        hidden = F.relu(self.fc1(x))

        mean = self.fc21(hidden)
        log_std = self.fc22(hidden)

        # [B, latent_dim * 2]
        return torch.cat([mean, log_std], dim=-1)


class MNISTEncoder1(nn.Module):
    """Simple MNIST encoder with one hidden layer, from MVAE paper"""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, latent_dim)
        self.fc32 = nn.Linear(512, latent_dim)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x.view(-1, 784)))
        h = self.swish(self.fc2(h))

        mean = self.fc31(h)
        log_std = self.fc32(h)

        # [B, latent_dim * 2]
        return torch.cat([mean, log_std], dim=-1)


class DeepMNISTEncoder(nn.Module):
    """Deeper MNIST encoder with one hidden layer, from MVAE paper"""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(784, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
        )

        self.fc31 = nn.Linear(512, latent_dim)
        self.fc32 = nn.Linear(512, latent_dim)

    def forward(self, x):
        h = self.enc(x.view(-1, 784))

        mean = self.fc31(h)
        log_std = self.fc32(h)

        # [B, latent_dim * 2]
        return torch.cat([mean, log_std], dim=-1)


class MNISTDecoder2(nn.Module):
    """Simple MNIST decoder with one hidden layer, from MMVAE paper"""

    def __init__(self, latent_dim):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, 400)
        self.fc2 = nn.Linear(400, 784)

    def forward(self, z: torch.tensor) -> torch.tensor:
        hidden = F.relu(self.fc1(z))
        output = self.fc2(hidden)

        # FIXME Have to reshape?
        return output.view(-1, 1, 28, 28)


class MNISTDecoder1(nn.Module):
    """Simple MNIST decoder with one hidden layer, from MVAE paper"""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 784)
        self.swish = Swish()

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))

        return self.fc4(h).view(-1, 1, 28, 28)


class LabelEncoder(nn.Module):
    """Label encoder for MNIST, from MVAE paper"""

    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Embedding(10, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, latent_dim)
        self.fc32 = nn.Linear(512, latent_dim)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x))
        h = self.swish(self.fc2(h))

        mean = self.fc31(h)
        log_std = self.fc32(h)

        # [B, latent_dim * 2]
        return torch.cat([mean, log_std], dim=-1)


class LabelDecoder(nn.Module):
    """Label decoder for MNIST, from MVAE paper"""

    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 10)
        self.swish = Swish()

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))

        return self.fc4(h)


class SVHNEncoder(nn.Module):
    """From MMVAE paper"""

    def __init__(self, latent_dim):
        super().__init__()

        n_filters = 32

        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(3, n_filters, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
        )
        self.c1 = nn.Conv2d(n_filters * 4, latent_dim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(n_filters * 4, latent_dim, 4, 1, 0, bias=True)
        # c1, c2 size: latent_dim x 1 x 1

    def forward(self, x):
        e = self.enc(x)
        mean = self.c1(e).squeeze()
        log_std = self.c2(e).squeeze()

        # [B, latent_dim * 2]
        return torch.cat([mean, log_std], dim=-1)


class DeepSVHNEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        n_filters = 32

        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(3, n_filters, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(n_filters * 2, n_filters * 2, 4, 1, 2, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
        )
        self.c1 = nn.Conv2d(n_filters * 4, latent_dim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(n_filters * 4, latent_dim, 4, 1, 0, bias=True)
        # c1, c2 size: latent_dim x 1 x 1

    def forward(self, x):
        e = self.enc(x)
        mean = self.c1(e).squeeze()
        log_std = self.c2(e).squeeze()

        # [B, latent_dim * 2]
        return torch.cat([mean, log_std], dim=-1)


class SVHNDecoder(nn.Module):
    """ Generate a SVHN image given a sample from the latent space. """

    def __init__(self, latent_dim):
        super().__init__()

        n_filters = 32

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, n_filters * 4, 4, 1, 0, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
            nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.ConvTranspose2d(n_filters, 3, 4, 2, 1, bias=True),
            # Output size: 3 x 32 x 32
        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        out = out.view(*z.size()[:-3], *out.size()[1:])

        return out


class PartitionedMNISTEncoder(nn.Module):
    def __init__(self, m_latent_dim: int, s_latent_dim: int):
        """MNIST encoder with partitioned latent space

        Parameters
        ----------
        m_latent_dim : int
            Size of modality-specific latent space
        s_latent_dim : int
            Size of modality-invariant (shared) latent space
        """
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)

        self.fc31 = nn.Linear(512, m_latent_dim)
        self.fc32 = nn.Linear(512, m_latent_dim)

        self.fc41 = nn.Linear(512, s_latent_dim)
        self.fc42 = nn.Linear(512, s_latent_dim)

        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x.view(-1, 784)))
        h = self.swish(self.fc2(h))

        # Modality-specific hidden vector
        m_mean = self.fc31(h)
        m_log_std = self.fc32(h)

        # Modality-invariant hidden vector
        s_mean = self.fc41(h)
        s_log_std = self.fc42(h)

        # [B, latent_dim * 2]
        return {
            "m": torch.cat([m_mean, m_log_std], dim=-1),
            "s": torch.cat([s_mean, s_log_std], dim=-1),
        }


class PartitionedMNISTEncoderV2(nn.Module):
    def __init__(self, m_latent_dim: int, s_latent_dim: int):
        """MNIST encoder with partitioned latent space

        Deeper layers for modality-invariant latent vector

        Parameters
        ----------
        m_latent_dim : int
            Size of modality-specific latent space
        s_latent_dim : int
            Size of modality-invariant (shared) latent space
        """
        super().__init__()
        # Encoding network
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)

        # Additional two processing layers
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)

        # Parameters for modality-specific latent
        self.fc_m = nn.Linear(512, m_latent_dim * 2)

        # Parameters for modality-invariant latent
        self.fc_s = nn.Linear(256, s_latent_dim * 2)

        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x.view(-1, 784)))
        h = self.swish(self.fc2(h))

        # Modality-specific hidden vector
        m_feature = self.fc_m(h)

        h = self.swish(self.fc3(h))
        h = self.swish(self.fc4(h))

        # Modality-invariant hidden vector
        s_feature = self.fc_s(h)

        # [B, latent_dim * 2]
        return {"m": m_feature, "s": s_feature}


class PartitionedSVHNEncoder(nn.Module):
    def __init__(self, m_latent_dim: int, s_latent_dim: int):
        """SVHN encoder with partitioned latent space

        Parameters
        ----------
        m_latent_dim : int
            Size of modality-specific latent space
        s_latent_dim : int
            Size of modality-invariant (shared) latent space
        """
        super().__init__()

        n_filters = 32

        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(3, n_filters, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
        )
        self.c1 = nn.Conv2d(n_filters * 4, m_latent_dim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(n_filters * 4, m_latent_dim, 4, 1, 0, bias=True)
        # c1, c2 size: latent_dim x 1 x 1

        self.c3 = nn.Conv2d(n_filters * 4, s_latent_dim, 4, 1, 0, bias=True)
        self.c4 = nn.Conv2d(n_filters * 4, s_latent_dim, 4, 1, 0, bias=True)

    def forward(self, x):
        e = self.enc(x)

        m_mean = self.c1(e).squeeze()
        m_log_std = self.c2(e).squeeze()

        s_mean = self.c3(e).squeeze()
        s_log_std = self.c4(e).squeeze()

        # [B, latent_dim * 2]
        return {
            "m": torch.cat([m_mean, m_log_std], dim=-1),
            "s": torch.cat([s_mean, s_log_std], dim=-1),
        }


class PartitionedSVHNEncoderV2(nn.Module):
    def __init__(self, m_latent_dim: int, s_latent_dim: int):
        """SVHN encoder with partitioned latent space

        Deeper layers for modality-invariant latent vector

        Parameters
        ----------
        m_latent_dim : int
            Size of modality-specific latent space
        s_latent_dim : int
            Size of modality-invariant (shared) latent space
        """
        super().__init__()

        n_filters = 32

        self.enc_1 = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(3, n_filters, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
        )

        self.enc_2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
        )

        self.c_m = nn.Linear(2048, m_latent_dim * 2)

        self.c_s = nn.Linear(512, s_latent_dim * 2)

    def forward(self, x):
        e = self.enc_1(x)
        e = e.view(e.shape[0], -1)

        m_feature = self.c_m(e)

        e = self.enc_2(e)

        s_feature = self.c_s(e)

        # [B, latent_dim * 2]
        return {"m": m_feature, "s": s_feature}
