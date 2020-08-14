"""To minimize KL(Q | P)
"""
import torch
import numpy as np


def f_phi(z, q_dist, p_dist):
    """f(z) = log q(z) - log p(z)"""
    return q_dist.log_prob(z).sum() - p_dist.log_prob(z).sum()


def reparam_kl(q_model, p_dist):
    q_dist = q_model.get_dist()
    z = q_dist.rsample()
    f = f_phi(z, q_dist, p_dist)

    return f


def reinforce_kl(q_model, p_dist):
    q_dist = q_model.get_dist()
    # Don't reparameterize
    z = q_dist.sample()

    f = f_phi(z, q_dist, p_dist)

    return q_dist.log_prob(z).sum() * f.detach() + f


def langevin_step(z, q_dist, eta=0.001, create_graph=False):
    assert z.requires_grad, "Taking gradients w.r.t. z"

    noise = torch.randn_like(z)
    log_q = q_dist.log_prob(z).sum()
    grad = torch.autograd.grad(log_q, z, create_graph=create_graph)[0]

    return z + eta * grad + np.sqrt(2 * eta) * noise


def langevin_kl(q_model, p_dist, s=10, t=1, eta=0.001):
    q_dist = q_model.get_dist()
    # Get cached sample
    z = q_model.prev_sample
    z.requires_grad_()

    # Run chain to get approx. sample (don't keep computational graph)
    for _ in range(s):
        z = langevin_step(z, q_dist, eta=eta, create_graph=False)

    z = z.detach()
    z.requires_grad_()

    # Run chain w/ reparameterization
    for _ in range(t):
        z = langevin_step(z, q_dist, eta=eta, create_graph=True)

    # Cache the sample
    q_model.prev_sample = z.detach()

    f = f_phi(z, q_dist, p_dist)

    return f
