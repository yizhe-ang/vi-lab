from typing import List
import torch.nn as nn
from nflows.distributions import Distribution
from .vae import VAE


class MultimodalVAE(VAE):
    def __init__(
        self,
        prior: Distribution,
        approximate_posterior: Distribution,
        likelihood: List[Distribution],
        inputs_encoder: nn.Module = None,
    ) -> None:
        super().__init__(prior, approximate_posterior, likelihood, inputs_encoder)

        # List of distributions
        self.likelihood = nn.ModuleList(self.likelihood)

        # FIXME To define helper methods
