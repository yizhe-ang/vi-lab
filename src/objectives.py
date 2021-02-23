from typing import List, Optional, Any, Dict
import torch
import torch.nn as nn
from nflows.utils import torchutils

# HELPERS ######################################################################


# def compute_multimodal_elbo(
#     model: nn.Module,
#     inputs: List[Optional[torch.Tensor]],
#     unimodal_q_contexts: List[Any],
#     likelihood_weights=None,
#     kl_multiplier=1.0,
# ):
#     """Computes multimodal objective terms in `vaevae_elbo`

#     Multimodal reconstruction term
#     + multimodal <-> unimodal posterior regularization terms
#     """
#     num_samples = 1

#     # Compute log prob of latents under the multimodal posterior
#     log_q_z_multi, latents, _ = model.log_q_z_x(inputs, num_samples=num_samples)

#     elbo = torch.zeros_like(log_q_z_multi)

#     # Compute log prob of latents under the unimodal posteriors
#     for q_context in unimodal_q_contexts:
#         log_q_z_uni = model.log_q_z_x(latent=latents, context=q_context)
#         # Compute multimodal <-> unimodal posterior regularization term
#         kl = log_q_z_multi - log_q_z_uni

#         elbo -= kl_multiplier * kl

#     # Compute likelihood term for all modalities
#     # Weight for each likelihood term
#     weights = likelihood_weights if likelihood_weights else [1.0] * len(inputs)
#     log_p_x_z = model.log_p_x_z(inputs, latents, weights, num_samples=num_samples)

#     elbo += log_p_x_z

#     return elbo


def compute_multimodal_elbo(
    model: nn.Module,
    inputs: List[Optional[torch.Tensor]],
    unimodal_q_contexts: List[Any] = None,
    keep_kl=True,
    likelihood_weights=None,
    num_samples=1,
    kl_multiplier=1.0,
    keepdim=False,
):
    """Computes unimodal or multimodal ELBO.

    Also returns posterior context / parameters
    """
    # Compute log prob of latents under the posterior
    log_q_z_x, latents, q_context = model.log_q_z_x(inputs, num_samples=num_samples)

    elbo = torch.zeros_like(log_q_z_x)

    # Compute unimodal <-> multimodal posterior regularization terms
    if unimodal_q_contexts:
        for context in unimodal_q_contexts:
            # Compute log prob of latents under unimodal posteriors
            log_q_z_uni = model.log_q_z_x(latent=latents, context=context)
            # Compute multimodal <-> unimodal posterior regularization term
            kl = log_q_z_x - log_q_z_uni

            elbo -= kl_multiplier * kl

    # Keep kl term
    if keep_kl:
        # Compute log prob of latents under the prior
        log_p_z = model.log_p_z(latents)

        elbo += kl_multiplier * (log_p_z - log_q_z_x)

    # Compute log prob of inputs under the decoder
    # Weight for each likelihood term
    weights = likelihood_weights if likelihood_weights else [1.0] * len(inputs)
    log_p_x_z = model.log_p_x_z(inputs, latents, weights, num_samples=num_samples)
    elbo += log_p_x_z

    elbo = torchutils.split_leading_dim(elbo, [-1, num_samples])
    if not keepdim:
        elbo = elbo.mean(1)  # Average ELBO across samples

    return elbo, q_context


# def compute_elbo(
#     model: nn.Module,
#     inputs: List[Optional[torch.Tensor]],
#     likelihood_weights=None,
#     num_samples=1,
#     kl_multiplier=1.0,
#     keepdim=False,
# ):
#     """Computes unimodal or multimodal ELBO.

#     Also returns posterior context / parameters
#     """
#     # Compute log prob of latents under the posterior
#     log_q_z_x, latents, q_context = model.log_q_z_x(inputs, num_samples=num_samples)

#     # Compute log prob of latents under the prior
#     log_p_z = model.log_p_z(latents)

#     # Compute log prob of inputs under the decoder
#     # Weight for each likelihood term
#     weights = likelihood_weights if likelihood_weights else [1.0] * len(inputs)
#     log_p_x_z = model.log_p_x_z(inputs, latents, weights, num_samples=num_samples)

#     # Compute ELBO
#     elbo = log_p_x_z + (kl_multiplier * (log_p_z - log_q_z_x))
#     elbo = torchutils.split_leading_dim(elbo, [-1, num_samples])
#     if not keepdim:
#         elbo = elbo.mean(1)  # Average ELBO across samples

#     return elbo, q_context


# OBJECTIVES ###################################################################


def stochastic_elbo(
    model: nn.Module,
    inputs: List[torch.Tensor],
    num_samples=1,
    keepdim=False,
):
    """Vanilla ELBO (no weights)."""
    # elbo, _ = compute_elbo(model, inputs, num_samples=num_samples, keepdim=keepdim)
    elbo, _ = compute_multimodal_elbo(
        model, inputs, num_samples=num_samples, keepdim=keepdim
    )

    return elbo


def mvae_elbo(
    model: nn.Module,
    batch: Dict[Any, Any],
    likelihood_weights=List[float],
    kl_multiplier=1.0,
) -> torch.Tensor:
    """ELBO(x1, x2) + ELBO(x1) + ELBO(x2)"""
    inputs = batch["data"]

    # To collate all elbo terms
    elbo_list = []

    # Compute unimodal / marginal elbos
    for i, x in enumerate(inputs):
        # Create input list (containing only one modality)
        xs = [None] * len(inputs)
        xs[i] = x

        elbo, _ = compute_multimodal_elbo(
            model,
            xs,
            likelihood_weights=likelihood_weights,
            kl_multiplier=kl_multiplier,
        )

        elbo_list.append(elbo)

    # Compute multimodal / joint elbo
    joint_elbo, _ = compute_multimodal_elbo(
        model,
        inputs,
        likelihood_weights=likelihood_weights,
        kl_multiplier=kl_multiplier,
    )
    elbo_list.append(joint_elbo)

    # Sum up all elbo terms
    return torch.stack(elbo_list).sum(0)


def vaevae_elbo(
    model: nn.Module,
    batch: Dict[Any, Any],
    likelihood_weights=List[float],
    kl_multiplier=1.0,
) -> torch.Tensor:
    """ELBO(x1) + ELBO(x2) + multimodal_recons + multimodal_reg"""
    inputs = batch["data"]
    paired = batch["paired"]

    # To collate all elbo terms
    elbo_list = []
    # To cache unimodal posterior parameters (for computing multimodal terms)
    unimodal_q_contexts = []

    # Compute unimodal elbos
    for i, x in enumerate(inputs):
        # Create input list (containing only one modality)
        xs = [None] * len(inputs)
        xs[i] = x

        elbo, q_context = compute_multimodal_elbo(
            model,
            xs,
            likelihood_weights=likelihood_weights,
            kl_multiplier=kl_multiplier,
        )

        elbo_list.append(elbo)
        unimodal_q_contexts.append(q_context)

    # Compute multimodal elbo terms
    # Multimodal reconstruction term
    # + multimodal <-> unimodal posterior regularization terms
    multimodal_elbo, _ = compute_multimodal_elbo(
        model,
        inputs,
        unimodal_q_contexts=unimodal_q_contexts,
        keep_kl=False,
        likelihood_weights=likelihood_weights,
        kl_multiplier=kl_multiplier,
    )
    # If not paired, set multimodal elbo terms to zero
    multimodal_elbo[~paired] = 0

    elbo_list.append(multimodal_elbo)

    # Sum up all elbo terms
    return torch.stack(elbo_list).sum(0)


def all_elbo(
    model: nn.Module,
    batch: Dict[Any, Any],
    likelihood_weights=List[float],
    kl_multiplier=1.0,
) -> torch.Tensor:
    inputs = batch["data"]

    """ELBO(x1, x2) + ELBO(x1) + ELBO(x1) + multimodal_reg"""
    # To collate all elbo terms
    elbo_list = []
    # To cache unimodal posterior parameters (for computing multimodal terms)
    unimodal_q_contexts = []

    # Compute unimodal elbos
    for i, x in enumerate(inputs):
        # Create input list (containing only one modality)
        xs = [None] * len(inputs)
        xs[i] = x

        elbo, q_context = compute_multimodal_elbo(
            model,
            xs,
            likelihood_weights=likelihood_weights,
            kl_multiplier=kl_multiplier,
        )

        elbo_list.append(elbo)
        unimodal_q_contexts.append(q_context)

    # Compute multimodal elbo terms
    # Multimodal reconstruction term
    # + multimodal <-> unimodal posterior regularization terms
    multimodal_elbo, _ = compute_multimodal_elbo(
        model,
        inputs,
        unimodal_q_contexts=unimodal_q_contexts,
        likelihood_weights=likelihood_weights,
        kl_multiplier=kl_multiplier,
    )
    elbo_list.append(multimodal_elbo)

    # Sum up all elbo terms
    return torch.stack(elbo_list).sum(0)


def jmvae_elbo(
    model: nn.Module,
    batch: Dict[Any, Any],
    likelihood_weights=List[float],
    kl_multiplier=1.0,
) -> torch.Tensor:
    inputs = batch["data"]

    """ELBO(x1, x2) + multimodal_reg"""
    # To cache unimodal posterior parameters (for computing multimodal terms)
    unimodal_q_contexts = []

    # Compute unimodal posterior contexts
    for i, x in enumerate(inputs):
        # Create input list (containing only one modality)
        xs = [None] * len(inputs)
        xs[i] = x

        # FIXME Wasted computation
        _, _, q_context = model.log_q_z_x(xs, num_samples=1)

        unimodal_q_contexts.append(q_context)

    # Compute multimodal elbo terms
    # Multimodal reconstruction term
    # + multimodal <-> unimodal posterior regularization terms
    elbo, _ = compute_multimodal_elbo(
        model,
        inputs,
        unimodal_q_contexts=unimodal_q_contexts,
        likelihood_weights=likelihood_weights,
        kl_multiplier=kl_multiplier,
    )

    return elbo
