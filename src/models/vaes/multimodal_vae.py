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
        torch.Tensor
            [n_modalities, B, D]
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

        return torch.stack(samples_list)
