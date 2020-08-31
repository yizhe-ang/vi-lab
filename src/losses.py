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
from torch.distributions import kl_divergence, Normal
from src.utils import log_mean_exp


def elbo_loss(vae, x, indices):
    _, log_px_z, _, qz_x, pz = vae(x)

    # Reconstruction term
    recon_loss = -log_px_z

    # Exact KL term
    kl_div = kl_divergence(qz_x, pz).sum(-1)

    # ELBO loss
    loss = recon_loss + kl_div

    # Take average across batch
    loss = loss.mean()
    recon_loss = recon_loss.mean()
    kl_div = kl_div.mean()

    return loss, recon_loss, kl_div

def elbo_loss_mc(vae, x, indices):
    log_qz_x, log_px_z, log_pz, _, _ = vae(x, indices=indices)

    # Reconstruction term
    recon_loss = -log_px_z

    # KL term (using monte carlo sampling)
    kl_div = (log_qz_x - log_pz).sum(-1)

    # ELBO loss
    loss = recon_loss + kl_div

    # Take average across batch
    loss = loss.mean()
    recon_loss = recon_loss.mean()
    kl_div = kl_div.mean()

    return loss, recon_loss, kl_div

def iwae_loss(vae, x, K):
    # FIXME Need to double-check
    log_qz_x, log_px_z, log_pz, _, _  = vae(x, K=K)

    lw = log_pz + log_px_z - log_qz_x

    iwae = log_mean_exp(lw).sum()

    # FIXME dummy values
    return -iwae, torch.tensor(0.), torch.tensor(0.)
