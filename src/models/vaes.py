"""
- Encoder
- Decoder
- Prior Dist
"""
import torch
import torch.nn as nn
from nflows.utils import torchutils
from nflows.distributions import Distribution


class VariationalAutoencoder(nn.Module):
    """Implementation of a standard VAE."""

    def __init__(
        self,
        prior: Distribution,
        approximate_posterior: Distribution,
        likelihood: Distribution,
        inputs_encoder: nn.Module = None,
    ) -> None:
        """
        Parameters
        ----------
        prior : Distribution
            p(z)
        approximate_posterior : Distribution
            q(z|x)
        likelihood : Distribution
            p(x|z)
        inputs_encoder : nn.Module, optional
            Required by flow models for the approx posterior;
            an encoder that encodes the input into a context vector,
            which is fed into both the cond base dist and each flow step,
            by default None
        """
        super().__init__()
        self.prior = prior
        self.approximate_posterior = approximate_posterior
        self.likelihood = likelihood
        self.inputs_encoder = inputs_encoder

    def forward(self, *args):
        raise RuntimeError("Forward method cannot be called for a VAE object.")

    def stochastic_elbo(
        self, inputs: torch.Tensor, num_samples=1, kl_multiplier=1, keepdim=False
    ) -> torch.Tensor:
        """Calculates an unbiased Monte-Carlo estimate of the evidence lower bound.

        Note: the KL term is also estimated via Monte Carlo

        Parameters
        ----------
        inputs : torch.Tensor
            [B, D]
        num_samples : int, optional
            Number of samples to use for the Monte-Carlo estimate, by default 1
        kl_multiplier : int, optional
            , by default 1
        keepdim : bool, optional
            , by default False

        Returns
        -------
        torch.Tensor
            An ELBO estimate for each input
            [B, K, D] if keepdim
            [B] otherwise
        """
        # Sample latents and calculate their log prob under the encoder
        if self.inputs_encoder is None:
            posterior_context = inputs
        else:
            posterior_context = self.inputs_encoder(inputs)

        latents, log_q_z = self.approximate_posterior.sample_and_log_prob(
            num_samples, context=posterior_context
        )
        latents = torchutils.merge_leading_dims(latents, num_dims=2)
        log_q_z = torchutils.merge_leading_dims(log_q_z, num_dims=2)

        # Compute log prob of latents under the prior
        log_p_z = self.prior.log_prob(latents)

        # Compute log prob of inputs under the decoder,
        inputs = torchutils.repeat_rows(inputs, num_reps=num_samples)
        log_p_x = self.likelihood.log_prob(inputs, context=latents)

        # Compute ELBO
        elbo = log_p_x + kl_multiplier * (log_p_z - log_q_z)
        elbo = torchutils.split_leading_dim(elbo, [-1, num_samples])

        if keepdim:
            return elbo
        else:
            return torch.sum(elbo, dim=1) / num_samples  # Average ELBO across samples

    # FIXME What is this exactly? IWAE bound?
    def log_prob_lower_bound(
        self, inputs: torch.Tensor, num_samples=100
    ) -> torch.Tensor:
        elbo = self.stochastic_elbo(inputs, num_samples=num_samples, keepdim=True)
        log_prob_lower_bound = torch.logsumexp(elbo, dim=1) - torch.log(
            torch.Tensor([num_samples])
        )

        return log_prob_lower_bound

    def decode(self, latents: torch.Tensor, mean: bool) -> torch.Tensor:
        """x ~ p(x|z)

        Parameters
        ----------
        latents : torch.Tensor
            [B, Z]
        mean : bool
            Uses the mean of the decoder instead of sampling from it

        Returns
        -------
        torch.Tensor
            [B, D]
        """
        if mean:
            return self.likelihood.mean(context=latents)
        else:
            samples = self.likelihood.sample(num_samples=1, context=latents)

            return torchutils.merge_leading_dims(samples, num_dims=2)

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
        """
        latents = self.prior.sample(num_samples)

        return self.decode(latents, mean)

    def encode(self, inputs: torch.Tensor, num_samples: int = None) -> torch.Tensor:
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
        if num_samples is None:
            latents = self.approximate_posterior.sample(num_samples=1, context=inputs)
            latents = torchutils.merge_leading_dims(latents, num_dims=2)
        else:
            latents = self.approximate_posterior.sample(
                num_samples=num_samples, context=inputs
            )

        return latents

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
