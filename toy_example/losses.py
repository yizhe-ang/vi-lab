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


def reinforce_kl_test(q_model, p_dist):
    q_dist = q_model.get_dist()
    # Don't reparameterize
    z = q_dist.sample()

    f = f_phi(z, q_dist, p_dist)

    return q_dist.log_prob(z).sum() * f.detach()


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


def gibbs_step(z, q_dist):
    """
    https://faculty.wcas.northwestern.edu/~lchrist/course/Korea_2016/note_on_Gibbs_sampling.pdf
    """
    # FIXME Make sure computaitonal graph for gradients is correctly recorded
    mu_1 = q_dist.loc[0]
    mu_2 = q_dist.loc[1]

    covariance_matrix = q_dist.covariance_matrix
    sigma_11 = covariance_matrix[0, 0]
    sigma_22 = covariance_matrix[1, 1]
    sigma_12 = covariance_matrix[0, 1]

    z2 = z[1]

    # p(z1 | z2)
    z1_loc = mu_1 + (sigma_12/sigma_22)*(z2 - mu_2)
    z1_var = (sigma_11*sigma_22 - sigma_12.square())/sigma_22

    z1 = z1_loc + z1_var.sqrt()*torch.randn(1)

    # p(z2 | z1)
    z2_loc = mu_2 + (sigma_12/sigma_11)*(z1 - mu_1)
    z2_var = (sigma_11*sigma_22 - sigma_12.square())/sigma_11

    z2 = z2_loc + z2_var.sqrt()*torch.randn(1)

    return torch.cat((z1, z2))


def gibbs_kl(q_model, p_dist, s=10, t=1):
    q_dist = q_model.get_dist()
    # Get cached sample
    # z = q_model.prev_sample

    # Or sample directly from dist?
    z = q_dist.sample()

    # Run w/o reparam
    with torch.no_grad():
        for _ in range(s):
            z = gibbs_step(z, q_dist)

    # Run w/ reparam
    for _ in range(t):
        z = gibbs_step(z, q_dist)

    # Cache the sample
    q_model.prev_sample = z.detach()

    f = f_phi(z, q_dist, p_dist)

    return f
