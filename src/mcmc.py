import numpy as np
import torch
from torch.distributions import Distribution


class LangevinMCMC:
    def __init__(self, dist: Distribution, eps: float) -> None:
        """Constructs Langevin MCMC sampler

        Parameters
        ----------
        distribution : Distribution
            Distribution from which we sample
        """
        self.dist = dist
        self.eps = eps

    def _step(self, x: torch.tensor, create_graph: bool) -> torch.tensor:
        """One step of MCMC, q(x'|x)

        Parameters
        ----------
        x : torch.tensor
            [B, D]
        create_graph : bool
            Whether to keep computational graph (for further gradient calculation)

        Returns
        -------
        torch.tensor
            [B, D], Next MCMC sample
        """
        return (
            x
            + self.eps * self._grad_log_density(x, create_graph)
            + np.sqrt(2 * self.eps) * torch.randn_like(x)
        )

    def _grad_log_density(self, x: torch.Tensor, create_graph: bool) -> torch.Tensor:
        """Computes vectorized gradient \nabla_x \log \pi(x)

        Parameters
        ----------
        x : torch.Tensor
            [B, D]
        create_graph : bool
            Whether to keep computational graph (for further gradient calculation)

        Returns
        -------
        torch.Tensor
            [B, D], Gradients of log density
        """
        assert x.requires_grad, "Taking gradients w.r.t. z"

        log_p = self.dist.log_prob(x).sum()
        grad = torch.autograd.grad(log_p, x, create_graph=create_graph)[0]

        return grad

    def simulate(
        self, x: torch.tensor, n_steps: int, create_graph: bool, save_samples=False
    ) -> torch.Tensor:
        """Run simulations for `n_steps` from a batch of data points

        Parameters
        ----------
        x : torch.tensor
            [B, D], Starting point for each chain
        n_steps : int
            Number of samples in Markov chain
        create_graph : bool
            Whether to keep computational graph (for further gradient calculation)

        Returns
        -------
        points : torch.tensor
            [B, D], Final sample from each chain
        """
        if save_samples:
            self.samples = torch.zeros(n_steps, *x.shape)

        x.requires_grad_()

        for i in range(n_steps):
            x = self._step(x, create_graph)

            if save_samples:
                self.samples[i] += x.detach()

        return x if create_graph else x.detach()
