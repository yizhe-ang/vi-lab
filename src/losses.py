"""Most objectives require:

Reconstruction Term
- Samples from q(z|x)
- log p(x|z)

KL Term
- KL[q(z|x) | p(z)]
- log q(z|x)
- log p(z)
"""
from torch.distributions import kl_divergence, Normal


def elbo_loss(vae, x):
    log_qz_x, z, qz_x = vae.encoder(x)
    pz = vae.prior(z)
    log_pz = pz.log_prob(z).sum(-1)

    # Reconstruction term
    log_px_z = vae.decoder(x, z)
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
