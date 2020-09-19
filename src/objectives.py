"""Most objectives require:

Reconstruction Term
- Samples from q(z|x)
- log p(x|z)

KL Term
- KL[q(z|x) | p(z)]
- log q(z|x)
- log p(z)

KL Annealing?

Pathwise derivative
Dreg
"""
from typing import List, Union
import torch
import torch.nn as nn
from src.utils import set_default_tensor_type
from nflows.utils import torchutils
from nflows.distributions import Distribution
from torch.distributions import Normal


# FIXME Test this function!!
# FIXME Any way to vectorize this?
@set_default_tensor_type(torch.cuda.FloatTensor)
def vaevae_elbo(
    model,
    inputs: List[torch.Tensor],
    likelihood_weights: List[float],
    num_samples=1,
    kl_multiplier=1.0,
    keepdim=False,
) -> torch.Tensor:
    # FIXME Add kl and likelihood weights?
    # FIXME Fix for keepdim
    # Compute kl analytically?

    x, y = inputs
    x_likelihood, y_likelihood = model.likelihood

    # Compute unimodal components
    elbo_sum, contexts = unimodal_elbos(model, inputs, likelihood_weights)
    # Parameters for q(z|x) and q(z|y)
    x_context, y_context = contexts

    # Compute bimodal components
    xy_context = model.inputs_encoder(inputs)

    z, log_q_z_xy = model.approximate_posterior.sample_and_log_prob(
        num_samples, context=xy_context
    )
    z = torchutils.merge_leading_dims(z, num_dims=2)
    log_q_z_xy = torchutils.merge_leading_dims(log_q_z_xy, num_dims=2)

    # HACK hardcode
    log_q_z_x = model.approximate_posterior.log_prob(z, context=x_context)
    log_q_z_y = model.approximate_posterior.log_prob(z, context=y_context)

    kl_1 = log_q_z_xy - log_q_z_x
    kl_2 = log_q_z_xy - log_q_z_y

    x = torchutils.repeat_rows(x, num_reps=num_samples)
    y = torchutils.repeat_rows(y, num_reps=num_samples)
    log_p_x_z = x_likelihood.log_prob(x, context=z)
    log_p_x_y = y_likelihood.log_prob(y, context=z)

    # FIXME Minus or plus KL??
    # FIXME How to weigh these terms?
    elbo = log_p_x_z + log_p_x_y - kl_1 - kl_2
    elbo_sum += elbo

    return elbo_sum


@set_default_tensor_type(torch.cuda.FloatTensor)
def stochastic_elbo(
    model: nn.Module,
    inputs: Union[torch.Tensor, List[torch.Tensor]],
    num_samples=1,
    kl_multiplier=1.0,
    likelihood_weight: Union[float, List[float]] = 1.0,
    keepdim=False,
) -> torch.Tensor:
    """Calculates an unbiased Monte-Carlo estimate of the evidence lower bound.

    Note: the KL term is also estimated via Monte Carlo

    Parameters
    ----------
    model : nn.Module
        VAE model
    inputs : torch.Tensor | List[torch.Tensor]
        [B, D], Supports multimodal inputs
    num_samples : int, optional
        Number of samples to use for the Monte-Carlo estimate, by default 1
    kl_multiplier : float, optional
        , by default 1.0
    likelihood_weight: float | List[float], optional
        How much to weigh the reconstruction term, by default 1.0
        Supports multimodal inputs
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
    if model.inputs_encoder is None:
        posterior_context = inputs
    else:
        posterior_context = model.inputs_encoder(inputs)

    latents, log_q_z = model.approximate_posterior.sample_and_log_prob(
        num_samples, context=posterior_context
    )
    latents = torchutils.merge_leading_dims(latents, num_dims=2)
    log_q_z = torchutils.merge_leading_dims(log_q_z, num_dims=2)

    # Compute log prob of latents under the prior
    log_p_z = model.prior.log_prob(latents)

    # Compute log prob of inputs under the decoder
    # If inputs are multimodal
    if isinstance(inputs, List):
        if likelihood_weight == 1.0:
            likelihood_weight = [1.0] * len(inputs)

        # Compute log prob of inputs under each decoder
        log_p_x = torch.zeros_like(log_p_z)

        for x, l, w in zip(inputs, model.likelihood, likelihood_weight):
            x = torchutils.repeat_rows(x, num_reps=num_samples)
            log_p_x += w * l.log_prob(x, context=latents)
    else:
        inputs = torchutils.repeat_rows(inputs, num_reps=num_samples)
        log_p_x = likelihood_weight * model.likelihood.log_prob(inputs, context=latents)

    # Compute ELBO
    elbo = log_p_x + (kl_multiplier * (log_p_z - log_q_z))
    elbo = torchutils.split_leading_dim(elbo, [-1, num_samples])

    if keepdim:
        return elbo
    else:
        return torch.sum(elbo, dim=1) / num_samples  # Average ELBO across samples


def unimodal_elbos(
    model: nn.Module,
    inputs: torch.Tensor,
    likelihood_weights=List[float],
    num_samples=1,
    kl_multiplier=1.0,
) -> torch.Tensor:

    batch_size = inputs[0].shape[0]

    # Compute the ELBO for each modality
    elbo_sum = torch.zeros(batch_size, device=inputs[0].device)
    # Cache the posterior context of each modality
    contexts = []

    for i, (x, likelihood, weight) in enumerate(
        zip(inputs, model.likelihood, likelihood_weights)
    ):
        unimodal_inputs = [None] * len(inputs)
        unimodal_inputs[i] = x

        posterior_context = model.inputs_encoder(unimodal_inputs)

        latents, log_q_z = model.approximate_posterior.sample_and_log_prob(
            num_samples, context=posterior_context
        )
        latents = torchutils.merge_leading_dims(latents, num_dims=2)
        log_q_z = torchutils.merge_leading_dims(log_q_z, num_dims=2)

        # Compute log prob of latents under the prior
        log_p_z = model.prior.log_prob(latents)

        # Compute log prob of inputs under the decoder
        x = torchutils.repeat_rows(x, num_reps=num_samples)
        log_p_x = likelihood.log_prob(x, context=latents)

        # Compute ELBO
        elbo = (weight * log_p_x) + (kl_multiplier * (log_p_z - log_q_z))
        elbo = torchutils.split_leading_dim(elbo, [-1, num_samples])

        # Average across # of samples
        elbo_sum += elbo.mean(dim=1)
        contexts.append(posterior_context)

    return elbo_sum, contexts


@set_default_tensor_type(torch.cuda.FloatTensor)
def log_prob_lower_bound(model, inputs: torch.Tensor, num_samples=100) -> torch.Tensor:
    # FIXME change stochastic_elbo to something else?
    elbo = stochastic_elbo(model, inputs, num_samples=num_samples, keepdim=True)
    log_prob_lower_bound = torch.logsumexp(elbo, dim=1) - torch.log(
        torch.Tensor([num_samples])
    )

    return log_prob_lower_bound


@set_default_tensor_type(torch.cuda.FloatTensor)
def path_derivative_elbo(
    model, inputs: torch.Tensor, num_samples=1, kl_multiplier=1, keepdim=False
):
    # Sample latents and calculate their log prob under the encoder

    # Get posterior mean and std parameters
    if model.inputs_encoder is None:
        posterior_context = inputs
    else:
        posterior_context = model.inputs_encoder(inputs)

    latents = model.approximate_posterior.sample(num_samples, context=posterior_context)
    latents = torchutils.merge_leading_dims(latents, num_dims=2)

    # Stop gradient on approx posterior parameters
    posterior_context_sg = posterior_context.detach()
    log_q_z = model.approximate_posterior.log_prob(
        latents, context=posterior_context_sg
    )

    # log_q_z = torchutils.merge_leading_dims(log_q_z, num_dims=2)

    # Compute log prob of latents under the prior
    log_p_z = model.prior.log_prob(latents)

    # Compute log prob of inputs under the decoder,
    inputs = torchutils.repeat_rows(inputs, num_reps=num_samples)
    log_p_x = model.likelihood.log_prob(inputs, context=latents)

    # Compute ELBO
    elbo = log_p_x + kl_multiplier * (log_p_z - log_q_z)
    elbo = torchutils.split_leading_dim(elbo, [-1, num_samples])

    if keepdim:
        return elbo
    else:
        return torch.sum(elbo, dim=1) / num_samples  # Average ELBO across samples


@set_default_tensor_type(torch.cuda.FloatTensor)
def langevin_elbo(
    model,
    inputs: torch.Tensor,
    cached_latents: torch.Tensor,
    num_samples=1,
    kl_multiplier=1,
    keepdim=False,
):
    # Sample latents and calculate their log prob under the encoder

    # Get posterior mean and std parameters
    if model.inputs_encoder is None:
        posterior_context = inputs
    else:
        posterior_context = model.inputs_encoder(inputs)

    latents = model.approximate_posterior._sample(
        num_samples, posterior_context, cached_latents
    )
    # latents = torchutils.merge_leading_dims(latents, num_dims=2)

    log_q_z = model.approximate_posterior.log_prob(latents, context=posterior_context)
    # means, log_stds = model.approximate_posterior._compute_params(posterior_context)
    # log_q_z = Normal(means, log_stds.exp()).log_prob(latents).sum(-1)
    with torch.no_grad():
        print(log_q_z.mean())

    # Compute log prob of latents under the prior
    log_p_z = model.prior.log_prob(latents)

    # Compute log prob of inputs under the decoder,
    inputs = torchutils.repeat_rows(inputs, num_reps=num_samples)
    log_p_x = model.likelihood.log_prob(inputs, context=latents)

    # Examine all components
    print(f"log q(z|x): {log_q_z.mean()}")
    print(f"log p(z): {log_p_z.mean()}")
    print(f"log p(x|z): {log_p_x.mean()}")

    # Compute ELBO
    elbo = log_p_x + kl_multiplier * (log_p_z - log_q_z)

    # Filter out bad samples
    # elbo = elbo[log_q_z < -10_000]

    elbo = torchutils.split_leading_dim(elbo, [-1, num_samples])

    if keepdim:
        return elbo, latents
    else:
        return (
            torch.sum(elbo, dim=1) / num_samples,
            latents,
        )  # Average ELBO across samples
