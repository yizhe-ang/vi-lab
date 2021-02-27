from typing import List, Optional

import torch
import torch.nn as nn
from nflows.distributions import Distribution
from nflows.utils import torchutils


class MultimodalVAE(nn.Module):
    def __init__(
        self,
        prior: Distribution,
        approximate_posterior: Distribution,
        likelihoods: List[Distribution],
        inputs_encoder: nn.Module,
    ) -> None:
        super().__init__()
        self.prior = prior
        self.approximate_posterior = approximate_posterior
        # List of decoder distributions
        self.likelihoods = nn.ModuleList(likelihoods)
        self.inputs_encoder = inputs_encoder

    def log_q_z_x(
        self,
        inputs: List[Optional[torch.Tensor]] = None,
        latent=None,
        context=None,
        num_samples=1,
    ):
        # If inputs not specified (and latent and context specified instead)
        if not inputs:
            return self._log_q_z_x(latent, context)

        # Compute posterior context / parameters
        q_context = self.inputs_encoder(inputs)

        # Compute posterior
        latent, log_prob = self.approximate_posterior.sample_and_log_prob(
            num_samples, context=q_context
        )
        latent = torchutils.merge_leading_dims(latent, num_dims=2)
        log_prob = torchutils.merge_leading_dims(log_prob, num_dims=2)

        # log_prob, sampled latents, posterior context / parameters
        return log_prob, latent, q_context

    def _log_q_z_x(self, latent, context):
        # Compute log_q_z_x with latent and context specified
        # Compute posterior
        log_prob = self.approximate_posterior.log_prob(latent, context=context)

        return log_prob

    def log_p_z(self, latents):
        log_prob = self.prior.log_prob(latents)

        return log_prob

    def log_p_x_z(self, inputs, latents, weights, num_samples=1):
        log_prob_list = []

        # Compute likelihood for each modality
        for x, likelihood, weight in zip(inputs, self.likelihoods, weights):
            # Account for missing modalities
            if x is None:
                continue

            x = torchutils.repeat_rows(x, num_reps=num_samples)
            log_prob_list.append(weight * likelihood.log_prob(x, context=latents))

        return torch.stack(log_prob_list).sum(0)

    def encode(
        self, inputs: torch.Tensor, num_samples: int = None
    ) -> torch.Tensor:
        """z ~ q(z|x)

        Parameters
        ----------
        inputs : torch.Tensor
            [B, D]
        num_samples : int, optional
            If None, only one latent sample is generated per input, by default None

        Returns
        -------
        torch.Tensor
            [B, Z] if num_samples is None,
            [B, K, Z] otherwise
        """
        if self.inputs_encoder is None:
            posterior_context = inputs
        else:
            posterior_context = self.inputs_encoder(inputs)

        if num_samples is None:
            latents = self.approximate_posterior.sample(
                num_samples=1, context=posterior_context
            )
            latents = torchutils.merge_leading_dims(latents, num_dims=2)
        else:
            latents = self.approximate_posterior.sample(
                num_samples=num_samples, context=posterior_context
            )

        return latents

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
        for l in self.likelihoods:
            if mean:
                samples = l.mean(context=latents)
            else:
                samples = l.sample(num_samples=1, context=latents)
                samples = torchutils.merge_leading_dims(samples, num_dims=2)

            samples_list.append(samples)

        return samples_list

    def sample(self, num_samples: int, mean=False) -> torch.Tensor:
        """z ~ p(z), x ~ p(x|z)

        Parameters
        ----------
        num_samples : int
        mean : bool, optional
            Uses the mean of the decoder instead of sampling from it, by default False

        Returns
        -------
        torch.Tensor
            [num_samples, D]
        List[torch.Tensor]
            List[num_samples, D] of length n_modalities
        """
        latents = self.prior.sample(num_samples)

        return self.decode(latents, mean)

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
