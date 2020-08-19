from base import Distribution, Proposal
import torch
from typing import Dict, Any
import numpy as np
import torch.distributions as dist


class MCMC:
    def __init__(self, distribution: Distribution, proposal: Proposal) -> None:
        """Constructs MCMC sampler

        Parameters
        ----------
        distribution : Distribution
            Distribution from which we sample
        proposal : Proposal
            MCMC proposal
        """
        self.distribution = distribution
        self.proposal = proposal

    def _step(self, x: torch.tensor, reject=True) -> torch.tensor:
        """One step of MCMC

        Parameters
        ----------
        x : torch.tensor
            [B, D]
        reject : bool, optional
            Whether to perform rejection step, by default True

        Returns
        -------
        torch.tensor
            [B, D], Next MCMC samples
        """
        batch_size = x.shape[0]

        x_prime = self.proposal.sample(x)

        # Apply rejection step
        acceptance_prob = (
            self.acceptance_prob(x_prime, x) if reject else torch.ones(batch_size)
        )
        mask = torch.rand(batch_size) < acceptance_prob
        x[mask] = x_prime[mask]

        # Keep track of # rejected for each parallel chain
        self._rejected += (1 - mask).type(torch.float32)

        return x

    def simulate(
        self, initial_point: torch.tensor, n_steps: int, n_parallel: int = 10
    ) -> Dict[str, Any]:
        """Run `n_parallel` simulations for `n_steps` starting from `initial_point`

        Parameters
        ----------
        initial_point : torch.tensor
            [D], Starting point for all chains
        n_steps : int
            Number of samples in Markov chain
        n_parallel : int, optional
            Number of parallel chains, by default 10

        Returns
        -------
        Dict[str, Any]
            points : torch.tensor
                [n_parallel, n_steps, D], Samples
            n_rejected : np.ndarray
                [n_parallel], Number of rejections for each chain
            rejection_rate : float
                Mean rejection rate over all chains
            means : torch.tensor
                [n_parallel, n_steps, D], means[c, s] = mean(points[c, :s])
            variances : torch.tensor
                [n_parallel, n_steps, D], variances[c, s, d] = variance(points[c, :s, d])
        """

        xs = []
        x = initial_point.repeat(n_parallel, 1)
        self._rejected = torch.zeros(n_parallel)

        dim = initial_point.shape[0]
        sums = np.zeros([n_parallel, dim])
        squares_sum = np.zeros([n_parallel, dim])

        means = []
        variances = []

        for i in range(n_steps):
            x = self._step(x)
            xs.append(x.numpy().copy())

            sums += xs[-1]
            squares_sum += xs[-1] ** 2

            mean, squares_mean = sums / (i + 1), squares_sum / (i + 1)
            means.append(mean.copy())
            variances.append(squares_mean - mean ** 2)

        xs = np.stack(xs, axis=1)
        means = np.stack(means, axis=1)
        variances = np.stack(variances, axis=1)

        return dict(
            points=xs,
            n_rejected=self._rejected.numpy(),
            rejection_rate=(self._rejected / n_steps).mean().item(),
            means=means,
            variances=variances,
        )

    def acceptance_prob(self, x_prime: torch.tensor, x: torch.tensor) -> torch.tensor:
        """Probability of acceptance \rho(x'|x)

        Parameters
        ----------
        x_prime : torch.tensor
            [B, D]
        x : torch.tensor
            [B, D]

        Returns
        -------
        torch.tensor
            [B], probability of acceptance for each point
        """
        pi_new = self.distribution.log_density(x_prime)
        pi_old = self.distribution.log_density(x)
        q_new = self.proposal.log_density(x_prime, x)
        q_old = self.proposal.log_density(x, x_prime)

        ratio = torch.exp(pi_new - pi_old + q_new - q_old)

        return ratio.clamp(0.0, 1.0)


class Langevin(Proposal):
    """Proposal given by,
    q(x'|x) = N(x' | x - 0.5*eps * \nabla \log \pi(x), eps)
    """

    def __init__(self, eps: float, dist: Distribution) -> None:
        """
        Parameters
        ----------
        eps : float
            Step size
        dist : Distribution
            Distribution sampling from
        """
        self.noise = dist.Normal(loc=0.0, scale=np.sqrt(eps))
        self.dist = dist
        self.eps = eps

    def sample(self, x: torch.tensor) -> torch.tensor:
        return (
            x
            - 0.5 * self.eps * self.dist.grad_log_density(x)
            + self.noise.sample(sample_shape=x.shape)
        )

    def log_density(self, x: torch.tensor, x_prime: torch.tensor) -> torch.tensor:
        xn = x - 0.5 * self.eps * self.dist.grad_log_density(x)

        # FIXME why self.noise instead of self.dist
        return self.noise.log_prob(x_prime - xn).sum(dim=-1)

    def __str__(self):
        return f"Langevin eps={self.eps}"
