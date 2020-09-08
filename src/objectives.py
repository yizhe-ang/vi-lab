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
import torch
from src.utils import set_default_tensor_type
from nflows.utils import torchutils


@set_default_tensor_type(torch.cuda.FloatTensor)
def stochastic_elbo(
    model, inputs: torch.Tensor, num_samples=1, kl_multiplier=1, keepdim=False
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

    log_q_z = model.approximate_posterior.log_prob(
        latents, context=posterior_context
    )
    with torch.no_grad():
        print(log_q_z.mean())

    # Compute log prob of latents under the prior
    log_p_z = model.prior.log_prob(latents)

    # Compute log prob of inputs under the decoder,
    inputs = torchutils.repeat_rows(inputs, num_reps=num_samples)
    log_p_x = model.likelihood.log_prob(inputs, context=latents)

    # Compute ELBO
    elbo = log_p_x + kl_multiplier * (log_p_z - log_q_z)
    elbo = torchutils.split_leading_dim(elbo, [-1, num_samples])

    if keepdim:
        return elbo, latents
    else:
        return (
            torch.sum(elbo, dim=1) / num_samples,
            latents,
        )  # Average ELBO across samples
