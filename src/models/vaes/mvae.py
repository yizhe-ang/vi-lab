from typing import List

import torch
import torch.nn as nn
from nflows.distributions import Distribution
from nflows.utils import torchutils

from src.utils import set_default_tensor_type

from .vae import VAE


class MultimodalVAE(VAE):
    def __init__(
        self,
        prior: Distribution,
        approximate_posterior: Distribution,
        likelihood: List[Distribution],
        inputs_encoder: nn.Module,
    ) -> None:
        super().__init__(prior, approximate_posterior, likelihood, inputs_encoder)

        # List of decoder distributions
        self.likelihood = nn.ModuleList(self.likelihood)

    def decode(self, latents: torch.Tensor, mean: bool) -> List[torch.Tensor]:
        """x ~ p(x|z) for each modality

        Parameters
        ----------
        latents : torch.Tensor
            [B, Z]
        mean : bool
            Uses the mean of the decoder instead of sampling from it

        Returns
        -------
        List[torch.Tensor]
            List[B, D] of length n_modalities
        """
        samples_list = []

        # Get samples from each decoder
        for l in self.likelihood:
            if mean:
                samples = l.mean(context=latents)
            else:
                samples = l.sample(num_samples=1, context=latents)
                samples = torchutils.merge_leading_dims(samples, num_dims=2)

            samples_list.append(samples)

        return samples_list

    @set_default_tensor_type(torch.cuda.FloatTensor)
    def cross_reconstruct(
        self, inputs: List[torch.Tensor], num_samples: int = None, mean=False
    ) -> torch.Tensor:
        """
        x -> z_x -> y,
        y -> z_y -> x

        Parameters
        ----------
        inputs : torch.Tensor
            [B, D]
        num_samples : int, optional
            Number of reconstructions to generate per input
            If None, only one reconstruction is generated per input,
            by default None
        mean : bool, optional
            Uses the mean of the decoder instead of sampling from it, by default False

        Returns
        -------
        torch.Tensor
            [B, D] if num_samples is None,
            [B, K, D] otherwise
        """
        # FIXME Only assuming two modalities
        x, y = inputs

        # x -> y
        x_latents = self.encode([x, None], num_samples)
        if num_samples is not None:
            x_latents = torchutils.merge_leading_dims(x_latents, num_dims=2)

        y_recons = self.decode(x_latents, mean)[1]
        if num_samples is not None:
            y_recons = torchutils.split_leading_dim(y_recons, [-1, num_samples])

        # y -> x
        y_latents = self.encode([None, y], num_samples)
        if num_samples is not None:
            y_latents = torchutils.merge_leading_dims(y_latents, num_dims=2)

        x_recons = self.decode(y_latents, mean)[0]
        if num_samples is not None:
            x_recons = torchutils.split_leading_dim(x_recons, [-1, num_samples])

        return [x_recons, y_recons]
