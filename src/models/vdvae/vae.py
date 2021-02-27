from typing import List, Union

import numpy as np
import torch
import torch.nn as nn

from .encoder_decoder import Decoder, Encoder


class VAE(nn.Module):
    def __init__(
        self,
        image_channels=3,
        enc_config: str = "32x5,32d2,16x3,o,16d2,8x3,8d2,4x3,o",
        dec_config: str = "o,4x2,8m4,8x3,o,16m8,16x5,32m16,32x10",
        width=256,
        zdim=16,
        bottleneck_multiple=0.25,
        scale_init_weights=False,
        no_bias_above=64,
        num_mixtures=10,
        image_size=32,
    ):
        super().__init__()
        self.encoder = Encoder(
            width,
            image_channels,
            enc_config,
            bottleneck_multiple,
            scale_init_weights,
        )
        self.decoder = Decoder(
            dec_config,
            width,
            zdim,
            bottleneck_multiple,
            scale_init_weights,
            no_bias_above,
            num_mixtures,
            image_size,
        )

    def forward(self, x, x_target):
        """
        Parameters
        ----------
        x : torch.Tensor
            Preprocessed input, [B, H, W, C]
        x_target : torch.Tensor
            Input processed for the loss (data.py -> preprocess_func)
            [B, H, W, C]

        Returns
        -------
        Dict[str, torch.Tensor]
            Scalar objective values
            {elbo, likelihood, kl}
        """
        # Compute activations from encoder
        activations = self.encoder.forward(x)
        # Final layer activations, and KL terms
        px_z, stats = self.decoder.forward(activations)

        # FIXME What do these terms actually mean? Why per pixel?
        # Likelihood term, [B]
        distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target)
        # KL term, [B]
        rate_per_pixel = torch.zeros_like(distortion_per_pixel)
        ndims = np.prod(x.shape[1:])
        for statdict in stats:
            rate_per_pixel += statdict["kl"].sum(dim=(1, 2, 3))
        rate_per_pixel /= ndims

        # Compute ELBO over the batch (the "loss" to call `backward` on)
        elbo = (distortion_per_pixel + rate_per_pixel).mean()
        return dict(
            elbo=elbo,
            distortion=distortion_per_pixel.mean(),
            rate=rate_per_pixel.mean(),
        )

    def forward_get_latents(self, x):
        """Compute inferred latents"""
        activations = self.encoder.forward(x)
        _, stats = self.decoder.forward(activations, get_latents=True)
        return stats

    def forward_uncond_samples(self, n_batch: int, t: Union[float, List[float]] = None):
        """Sample from generative model

        Parameters
        ----------
        n_batch : int
            Number of samples
        t : Union[float, List[float]], optional
            Temperature of sampling variance

        Returns
        -------
        torch.Tensor
            [B, H, W, C], uint8, [0, 255]
        """
        px_z = self.decoder.forward_uncond(n_batch, t=t)

        return self.decoder.out_net.sample(px_z)

    def forward_samples_set_latents(self, n_batch, latents, t=None):
        px_z = self.decoder.forward_manual_latents(n_batch, latents, t=t)
        return self.decoder.out_net.sample(px_z)


class MVAE(nn.Module):
    # Just take two decoders and line them up
    # Combine q's and p's using some fusion module (PoE or o.w.)
    def __init__(
        self,
        n_modalities=2,
        image_channels=3,
        enc_config: str = "32x5,32d2,16x3,o,16d2,8x3,8d2,4x3,o",
        dec_config: str = "o,4x2,8m4,8x3,o,16m8,16x5,32m16,32x10",
        width=256,
        zdim=16,
        bottleneck_multiple=0.25,
        scale_init_weights=False,
        no_bias_above=64,
        num_mixtures=10,
        image_size=32,
    ):
        super().__init__()
        self.shared_latent_idx = 0

        self.encoders = nn.ModuleList(
            [
                Encoder(
                    width,
                    image_channels,
                    enc_config,
                    bottleneck_multiple,
                    scale_init_weights,
                )
                for _ in range(n_modalities)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                Decoder(
                    dec_config,
                    width,
                    zdim,
                    bottleneck_multiple,
                    scale_init_weights,
                    no_bias_above,
                    num_mixtures,
                    image_size,
                )
                for _ in range(n_modalities)
            ]
        )

    def forward(self, x, x_target):
        # List of activations from each encoder
        activations = [encoder.forward(x) for encoder in self.encoders]