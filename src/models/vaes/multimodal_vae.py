from typing import List
import torch.nn as nn
from nflows.distributions import Distribution
from .vae import VAE
import torch
from nflows.utils import torchutils


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

    def decode(self, latents: torch.Tensor, mean: bool) -> torch.Tensor:
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

    def reconstruct(
        self, inputs: torch.Tensor, num_samples: int = None, mean=False
    ) -> torch.Tensor:
        """Reconstruct given inputs

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
            [B, K, Z] otherwise
        """
        latents = self.encode(inputs, num_samples)
        if num_samples is not None:
            latents = torchutils.merge_leading_dims(latents, num_dims=2)

        recons = self.decode(latents, mean)
        if num_samples is not None:
            recons = torchutils.split_leading_dim(recons, [-1, num_samples])

        return recons
