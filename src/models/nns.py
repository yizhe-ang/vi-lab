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
from typing import List, Optional, Tuple
from nflows.distributions import ConditionalDiagonalNormal


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""

    def forward(self, x):
        return x * F.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        resample=None,
        activation=F.relu,
        dropout_probability=0.0,
        first=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.resample = resample
        self.activation = activation

        self.residual_layer_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1
        )

        if resample is None:
            self.shortcut_layer = nn.Identity()
            self.residual_2_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
            )
        elif resample == "down":
            self.shortcut_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=2 * in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            self.residual_2_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=2 * in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        elif resample == "up":
            self.shortcut_layer = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0 if first else 1,
            )
            self.residual_2_layer = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0 if first else 1,
            )

        if dropout_probability > 0:
            self.dropout = nn.Dropout(dropout_probability)
        else:
            self.dropout = None

    def forward(self, inputs):

        shortcut = self.shortcut_layer(inputs)
        residual_1 = self.activation(inputs)
        residual_1 = self.residual_layer_1(residual_1)
        if self.dropout is not None:
            residual_1 = self.dropout(residual_1)
        residual_2 = self.activation(residual_1)
        residual_2 = self.residual_2_layer(residual_2)

        return shortcut + residual_2


class ConvEncoder(nn.Module):
    def __init__(
        self,
        context_features,
        channels_multiplier,
        activation=F.relu,
        dropout_probability=0.0,
    ):
        super().__init__()
        self.context_features = context_features
        self.channels_multiplier = channels_multiplier
        self.activation = activation

        self.initial_layer = nn.Conv2d(1, channels_multiplier, kernel_size=1)
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=channels_multiplier,
                    dropout_probability=dropout_probability,
                ),
                ResidualBlock(
                    in_channels=channels_multiplier,
                    resample="down",
                    dropout_probability=dropout_probability,
                ),
                ResidualBlock(
                    in_channels=channels_multiplier * 2,
                    dropout_probability=dropout_probability,
                ),
                ResidualBlock(
                    in_channels=channels_multiplier * 2,
                    resample="down",
                    dropout_probability=dropout_probability,
                ),
                ResidualBlock(
                    in_channels=channels_multiplier * 4,
                    dropout_probability=dropout_probability,
                ),
                ResidualBlock(
                    in_channels=channels_multiplier * 4,
                    resample="down",
                    dropout_probability=dropout_probability,
                ),
            ]
        )
        self.final_layer = nn.Linear(
            in_features=(4 * 4 * channels_multiplier * 8), out_features=context_features
        )

    def forward(self, inputs):
        temps = self.initial_layer(inputs)
        for residual_block in self.residual_blocks:
            temps = residual_block(temps)
        temps = self.activation(temps)
        outputs = self.final_layer(
            temps.reshape(-1, 4 * 4 * self.channels_multiplier * 8)
        )
        return outputs


class ConvDecoder(nn.Module):
    def __init__(
        self,
        latent_features,
        channels_multiplier,
        activation=F.relu,
        dropout_probability=0.0,
    ):
        super().__init__()
        self.latent_features = latent_features
        self.channels_multiplier = channels_multiplier
        self.activation = activation

        self.initial_layer = nn.Linear(
            in_features=latent_features, out_features=(4 * 4 * channels_multiplier * 8)
        )
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=channels_multiplier * 8,
                    dropout_probability=dropout_probability,
                ),
                ResidualBlock(
                    in_channels=channels_multiplier * 8,
                    resample="up",
                    first=True,
                    dropout_probability=dropout_probability,
                ),
                ResidualBlock(
                    in_channels=channels_multiplier * 4,
                    dropout_probability=dropout_probability,
                ),
                ResidualBlock(
                    in_channels=channels_multiplier * 4,
                    resample="up",
                    dropout_probability=dropout_probability,
                ),
                ResidualBlock(
                    in_channels=channels_multiplier * 2,
                    dropout_probability=dropout_probability,
                ),
                ResidualBlock(
                    in_channels=channels_multiplier * 2,
                    resample="up",
                    dropout_probability=dropout_probability,
                ),
            ]
        )
        self.final_layer = nn.Conv2d(
            in_channels=channels_multiplier, out_channels=1, kernel_size=1
        )

    def forward(self, inputs):
        temps = self.initial_layer(inputs).reshape(
            -1, self.channels_multiplier * 8, 4, 4
        )
        for residual_block in self.residual_blocks:
            temps = residual_block(temps)
        temps = self.activation(temps)
        outputs = self.final_layer(temps)
        return outputs


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


class ProductOfExpertsEncoder(nn.Module):
    def __init__(self, latent_dim: int, encoders: List[nn.Module]) -> None:
        """Composes multiple unimodal encoders into a multimodal encoder
        using Product of Experts (Gaussians).

        Parameters
        ----------
        latent_dim : int
        encoders : List[nn.Module]
            An encoder for each modality
        """
        super().__init__()

        # Init unimodal gaussian dists
        self.dists = nn.ModuleList(
            [
                ConditionalDiagonalNormal((latent_dim,), context_encoder=e)
                for e in encoders
            ]
        )

    def forward(self, xs: List[Optional[torch.Tensor]]):
        """
        Parameters
        ----------
        xs : List[Optional[torch.Tensor]]
            An input for each encoder. Allows for missing modalities.
            E.g. [x, y] or [x, None] or [None, y]
        """
        means = []
        log_stds = []

        assert len(self.dists) == len(
            xs
        ), "Number of encoders and inputs must be the same!"

        # Compute params for each unimodal dist
        for dist, x in zip(self.dists, xs):
            # Ignore for missing modalities
            if x is None:
                continue

            m, s = dist._compute_params(x)
            means.append(m)
            log_stds.append(s)

        # Add params of prior expert; assume prior is standard normal
        means.append(torch.zeros_like(means[0]))
        log_stds.append(torch.zeros_like(log_stds[0]))

        # Combine params using Product of Experts
        pd_means, pd_log_stds = self._product_of_experts(
            torch.stack(means), torch.stack(log_stds)
        )

        return torch.cat([pd_means, pd_log_stds], dim=-1)

    def _product_of_experts(
        self, means: torch.Tensor, log_stds: torch.Tensor, eps=1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return parameters for product of K independent experts.
        See https://arxiv.org/pdf/1410.7827.pdf for equations.

        Parameters
        ----------
        means : torch.Tensor
            [K, B, D]
        log_stds : torch.Tensor
            [K, B, D]
        eps : float, optional
            , by default 1e-8

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            means,  log_stds
            [B, D], [B, D]
        """
        # FIXME Double-check
        log_vars = log_stds * 2
        var = torch.exp(log_vars) + eps

        # precision of i-th Gaussian expert at point x
        T = 1.0 / (var + eps)
        pd_means = torch.sum(means * T, dim=0) / torch.sum(T, dim=0)
        pd_vars = 1.0 / torch.sum(T, dim=0)
        pd_log_stds = torch.log(pd_vars + eps) / 2

        return pd_means, pd_log_stds


def conv_encoder(n_outputs: int, dropout_prob=0.0):
    return ConvEncoder(
        context_features=n_outputs,
        channels_multiplier=16,
        dropout_probability=dropout_prob,
    )
