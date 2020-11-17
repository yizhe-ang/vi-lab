from typing import List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows import transforms
from nflows.distributions import (
    ConditionalDiagonalNormal,
    ConditionalIndependentBernoulli,
    Distribution,
    StandardNormal,
)
from nflows.flows import Flow
from nflows.nn.nets import ResidualNet
from nflows.transforms import Transform
from nflows.utils import torchutils
from torch.distributions import Categorical

from src.models.base import ConvDecoder, ConvEncoder, MLP

__all__ = [
    "standard_normal",
    "standard_flow",
    "standard_cond_flow",
    "cond_normal",
    "diagonal_normal",
    "cond_flow",
    "ConditionalIndependentBernoulli",
]


class ConditionalLangevinDiagonalNormal(ConditionalDiagonalNormal):
    def __init__(self, shape, s, t, eps, context_encoder=None):
        super().__init__(shape, context_encoder)

        self.s = s
        self.t = t
        self.eps = eps

    def _sample(self, num_samples, context, samples):
        """Sample using Langevin Dynamics instead"""
        context_size = context.shape[0]

        # Generate samples
        samples.requires_grad_()

        for _ in range(self.s):
            samples = self._step(samples, context, create_graph=False)
        samples = samples.detach()

        samples.requires_grad_()

        for _ in range(self.t):
            samples = self._step(samples, context, create_graph=True)

        return samples
        # return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _step(self, inputs, context, create_graph):
        return (
            inputs
            + self.eps * self._grad_log_prob(inputs, context, create_graph)
            + np.sqrt(2 * self.eps) * torch.randn_like(inputs)
        )

    def _grad_log_prob(self, inputs, context, create_graph):
        assert inputs.requires_grad, "Taking gradients w.r.t. inputs"

        log_p = self.log_prob(inputs, context).sum()
        grad = torch.autograd.grad(log_p, inputs, create_graph=create_graph)[0]

        return grad


# HELPER TRANSFORMS ############################################################
def create_lu_linear(n_dim: int) -> Transform:
    return transforms.CompositeTransform(
        [
            transforms.RandomPermutation(n_dim),
            transforms.LULinear(n_dim, identity_init=True),
        ]
    )


def create_rq_coupling(
    n_dim: int,
    step_index: int,
    context_features=None,
    n_bins=8,
    tail_bound=3,
    apply_unconditional_transform=True,
    # Hypernet args
    n_hidden=128,
    n_transform_blocks=2,
    dropout_prob=0.0,
    use_batch_norm=False,
) -> Transform:
    return transforms.PiecewiseRationalQuadraticCouplingTransform(
        mask=torchutils.create_alternating_binary_mask(
            features=n_dim, even=(step_index % 2 == 0)
        ),
        transform_net_create_fn=lambda in_features, out_features: ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=n_hidden,
            context_features=context_features,
            num_blocks=n_transform_blocks,
            activation=F.relu,
            dropout_probability=dropout_prob,
            use_batch_norm=use_batch_norm,
        ),
        num_bins=n_bins,
        tails="linear",
        tail_bound=tail_bound,
        apply_unconditional_transform=apply_unconditional_transform,
    )


def create_affine_coupling(
    n_dim: int,
    step_index: int,
    context_features=None,
    # Hypernet args
    n_hidden=128,
    n_transform_blocks=2,
    dropout_prob=0.0,
    use_batch_norm=False,
):
    return transforms.AffineCouplingTransform(
        mask=torchutils.create_alternating_binary_mask(
            features=n_dim, even=(step_index % 2 == 0)
        ),
        transform_net_create_fn=lambda in_features, out_features: ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=n_hidden,
            context_features=context_features,
            num_blocks=n_transform_blocks,
            activation=F.relu,
            dropout_probability=dropout_prob,
            use_batch_norm=use_batch_norm,
        ),
    )


create_flow_dict = {
    "rq_coupling": create_rq_coupling,
    "affine_coupling": create_affine_coupling,
}


# PRIORS #######################################################################
def standard_normal(n_dim: int) -> Distribution:
    return StandardNormal((n_dim,))


def standard_flow(
    n_dim: int, flow_type: str, n_flow_steps=10, n_flow_hidden=128, dropout_prob=0.0
) -> Distribution:
    """StandardNormal -> LU -> RQ (C) -> ... -> LU"""
    create_flow = create_flow_dict[flow_type]

    base_dist = StandardNormal((n_dim,))
    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    create_lu_linear(n_dim),
                    create_flow(
                        n_dim,
                        i,
                        n_hidden=n_flow_hidden,
                    ),
                ]
            )
            for i in range(n_flow_steps)
        ]
    )
    transform = transforms.CompositeTransform([transform, create_lu_linear(n_dim)])

    return Flow(transform, base_dist)


def standard_cond_flow(
    n_dim: int,
    context_features: int,
    flow_type: str,
    cond_base=False,
    n_flow_steps=10,
    n_flow_hidden=128,
    dropout_prob=0.0,
) -> Distribution:
    """Standard flow with layers conditioned on additional context vector"""
    create_flow = create_flow_dict[flow_type]

    if cond_base:
        context_encoder = nn.Linear(context_features, n_dim * 2)
        base_dist = ConditionalDiagonalNormal(
            shape=[n_dim], context_encoder=context_encoder
        )
    else:
        base_dist = StandardNormal((n_dim,))

    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    create_lu_linear(n_dim),
                    create_flow(
                        n_dim,
                        i,
                        context_features=context_features,
                        n_hidden=n_flow_hidden,
                    ),
                ]
            )
            for i in range(n_flow_steps)
        ]
    )
    transform = transforms.CompositeTransform([transform, create_lu_linear(n_dim)])

    return Flow(transform, base_dist)


def cond_normal(n_dim: int, context_features: int):
    context_encoder = MLP(
        context_features, n_dim * 2, hidden_units=[context_features, context_features]
    )
    return ConditionalDiagonalNormal(shape=[n_dim], context_encoder=context_encoder)


# APPROX POSTERIORS ############################################################
def diagonal_normal(n_dim: int) -> Distribution:
    return ConditionalDiagonalNormal(shape=[n_dim])


def cond_diagonal_normal(n_dim: int, dropout_prob=0.0) -> Distribution:
    context_encoder = ConvEncoder(
        context_features=n_dim * 2,
        channels_multiplier=16,
        dropout_probability=dropout_prob,
    )
    return ConditionalDiagonalNormal(shape=[n_dim], context_encoder=context_encoder)


def cond_langevin_diagonal_normal(
    n_dim: int, s: int, t: int, eps: float, dropout_prob=0.0
) -> Distribution:
    context_encoder = ConvEncoder(
        context_features=n_dim * 2,
        channels_multiplier=16,
        dropout_probability=dropout_prob,
    )
    return ConditionalLangevinDiagonalNormal(
        shape=[n_dim], s=s, t=t, eps=eps, context_encoder=context_encoder
    )


def cond_flow(
    n_dim: int, flow_type: str, n_flow_steps=10, n_flow_hidden=128, dropout_prob=0.0
) -> Distribution:
    create_flow = create_flow_dict[flow_type]

    context_features = n_dim * 2

    # To map context features into parameters for gaussian base dist
    context_encoder = nn.Linear(context_features, n_dim * 2)

    base_dist = ConditionalDiagonalNormal(
        shape=[n_dim], context_encoder=context_encoder
    )

    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    create_lu_linear(n_dim),
                    create_flow(
                        n_dim,
                        i,
                        context_features=context_features,
                        n_hidden=n_flow_hidden,
                    ),
                ]
            )
            for i in range(n_flow_steps)
        ]
    )
    transform = transforms.CompositeTransform([transform, create_lu_linear(n_dim)])

    # FIXME Why inverse?
    return Flow(transforms.InverseTransform(transform), base_dist)


# For PoE?
def flow(n_dim: int, n_flow_steps=10, dropout_prob=0.0) -> Distribution:

    base_dist = ConditionalDiagonalNormal(shape=[n_dim])

    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    create_lu_linear(n_dim),
                    create_rq_coupling(n_dim, i, context_features=n_dim * 2),
                ]
            )
            for i in range(n_flow_steps)
        ]
    )
    transform = transforms.CompositeTransform([transform, create_lu_linear(n_dim)])

    # FIXME Why inverse?
    return Flow(transforms.InverseTransform(transform), base_dist)


# LIKELIHOODS ##################################################################
class ConditionalCategorical(Distribution):
    def __init__(self, shape, context_encoder=None):
        """Constructor.
        Args:
            shape: list, tuple or torch.Size, the shape of the input variables.
            context_encoder: callable or None, encodes the context to the distribution parameters.
                If None, defaults to the identity function.
        """
        super().__init__()

        self._shape = torch.Size(shape)

        if context_encoder is None:
            self._context_encoder = lambda x: x
        else:
            self._context_encoder = context_encoder

    def _compute_params(self, context):
        """Compute the logits from context."""
        if context is None:
            raise ValueError("Context can't be None.")

        logits = self._context_encoder(context)
        if logits.shape[0] != context.shape[0]:
            raise RuntimeError(
                "The batch dimension of the parameters is inconsistent with the input."
            )

        return logits

    def _log_prob(self, inputs, context):
        # `inputs` should be a batch of labels, [B,]

        # Compute parameters.
        logits = self._compute_params(context)
        assert logits.shape[0] == inputs.shape[0]

        return Categorical(logits=logits).log_prob(inputs)

    def _sample(self, num_samples, context):
        # Compute parameters.
        logits = self._compute_params(context)

        samples = Categorical(logits=logits).sample(torch.Size([num_samples]))

        return samples.permute(1, 0)  # [B, K]

    def _mean(self, context):
        # FIXME
        # Return most probable class
        logits = self._compute_params(context)

        return torch.argmax(logits, dim=-1)


def cond_inpt_bernoulli(
    shape: List[int], latent_dim: int, decoder: nn.Module
) -> Distribution:
    """Returns a decoder with a Bernoulli distribution

    Parameters
    ----------
    shape : List[int]
        Output shape
    latent_dim : int
    decoder : nn.Module
        Decoder that maps from latent features to distribution parameters

    Returns
    -------
    Distribution
    """
    return ConditionalIndependentBernoulli(shape=shape, context_encoder=decoder)


def mnist_cond_indpt_bernoulli(n_dim: int, dropout_prob=0.0):
    latent_decoder = ConvDecoder(
        latent_features=n_dim,
        channels_multiplier=16,
        dropout_probability=dropout_prob,
    )

    return ConditionalIndependentBernoulli(
        shape=[1, 28, 28], context_encoder=latent_decoder
    )
