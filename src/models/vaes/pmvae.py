import torch
import torch.nn as nn
from nflows.distributions import Distribution
from typing import List, Dict, Any, Optional
from nflows.utils import torchutils


class PartitionedMultimodalVAE(nn.Module):
    def __init__(
        self,
        s_prior: Distribution,
        m_priors: List[Distribution],
        s_posterior: Distribution,
        m_posteriors: List[Distribution],
        likelihoods: List[Distribution],
        inputs_encoder: nn.Module,
    ):
        super().__init__()
        self.s_prior = s_prior
        self.m_priors = nn.ModuleList(m_priors)

        self.s_posterior = s_posterior
        self.m_posteriors = nn.ModuleList(m_posteriors)

        self.likelihoods = nn.ModuleList(likelihoods)
        self.inputs_encoder = inputs_encoder

    def decode(self, latents: Dict[Any, Any], mean: bool) -> List[torch.Tensor]:
        """x ~ p(x|z) for each modality

        Parameters
        ----------
        latents : Dict[Any, Any]
            {"m": m_latents,            "s": s_latent}
            {"m": List[Optional[B, Z]], "s": [B, Z]}
        mean : bool
            Uses the mean of the decoder instead of sampling from it

        Returns
        -------
        List[torch.Tensor]
            List[B, D] of length n_modalities
        """
        samples_list = []
        m_latents = latents["m"]
        s_latent = latents["s"]
        batch_size = s_latent.shape[0]

        # Get samples from each decoder
        for likelihood, prior, m_latent in zip(
            self.likelihoods, self.m_priors, m_latents
        ):
            # If missing m_latent, sample from prior instead
            if m_latent is None:
                m_latent = prior.sample(batch_size)

            # Concat modality-specific and -invariant latents
            concat_latent = torch.cat([m_latent, s_latent], dim=-1)

            if mean:
                samples = likelihood.mean(context=concat_latent)
            else:
                samples = likelihood.sample(num_samples=1, context=concat_latent)
                samples = torchutils.merge_leading_dims(samples, num_dims=2)

            samples_list.append(samples)

        return samples_list

    def encode(
        self, inputs: List[Optional[torch.Tensor]], num_samples: int = None
    ) -> Dict[Any, Any]:
        """Encode into modality-specific and -invariant latent space

        Parameters
        ----------
        inputs : List[Optional[torch.Tensor]]
        num_samples : int, optional

        Returns
        -------
        Dict[Any, Any]
            {"m": m_latents,               "s": s_latent}
            {"m": List[Optional[B, Z]],    "s": [B, Z]}
            {"m": List[Optional[B, K, Z]], "s": [B, K, Z]}
        """
        # Encode into posterior dist parameters
        posterior_context = self.inputs_encoder(inputs)
        m_contexts = posterior_context["m"]
        s_context = posterior_context["s"]

        if num_samples is None:
            # Account for missing modalities
            m_latents = [
                None
                if context is None
                else torchutils.merge_leading_dims(
                    posterior.sample(num_samples=1, context=context), num_dims=2
                )
                for posterior, context in zip(self.m_posteriors, m_contexts)
            ]

            s_latent = self.s_posterior.sample(num_samples=1, context=s_context)
            s_latent = torchutils.merge_leading_dims(s_latent, num_dims=2)

        else:
            m_latents = [
                None
                if context is None
                else posterior.sample(num_samples=num_samples, context=context)
                for posterior, context in zip(self.m_posteriors, m_contexts)
            ]

            s_latent = self.s_posterior.sample(
                num_samples=num_samples, context=s_context
            )

        return {"m": m_latents, "s": s_latent}

    def sample(self, num_samples: int, mean=False) -> List[torch.Tensor]:
        """z ~ p(z), x ~ p(x|z)

        Parameters
        ----------
        num_samples : int
        mean : bool, optional
            Uses the mean of the decoder instead of sampling from it, by default False

        Returns
        -------
        List[torch.Tensor]
            List[num_samples, D] of length n_modalities
        """
        m_latents = [prior.sample(num_samples) for prior in self.m_priors]
        s_latent = self.s_prior.sample(num_samples)

        return self.decode({"m": m_latents, "s": s_latent}, mean)

    def reconstruct(
        self, inputs: List[Optional[torch.Tensor]], num_samples: int = None, mean=False
    ) -> List[torch.Tensor]:
        pass

    def cross_reconstruct(
        self, inputs: List[torch.Tensor], num_samples: int = None, mean=False
    ) -> torch.Tensor:
        """
        x -> z_x -> y,
        y -> z_y -> x

        Parameters
        ----------
        inputs : List[torch.Tensor]
            List[B, D]
        num_samples : int, optional
            Number of reconstructions to generate per input
            If None, only one reconstruction is generated per input,
            by default None
        mean : bool, optional
            Uses the mean of the decoder instead of sampling from it, by default False

        Returns
        -------
        torch.Tensor
            [B, D] if num_samples is None,
            [B, K, D] otherwise
        """
        # FIXME Only assuming two modalities
        x, y = inputs

        # FIXME Only works for `num_samples` = None

        # x -> y
        x_latents = self.encode([x, None], num_samples)
        y_recons = self.decode(x_latents, mean)[1]

        # y -> x
        y_latents = self.encode([None, y], num_samples)
        x_recons = self.decode(y_latents, mean)[0]

        return [x_recons, y_recons]


class HierPMVAE_v1(PartitionedMultimodalVAE):
    def log_q_z(self, inputs: List[Optional[torch.Tensor]], num_samples: int = None):
        posterior_context = self.inputs_encoder(inputs)
        m_contexts = posterior_context["m"]
        s_context = posterior_context["s"]

        s_latent, log_q_z_s = self.s_posterior.sample_and_log_prob(
            num_samples, s_context
        )

        m_latents = []
        log_q_z_ms = []

        for posterior, context in zip(self.m_posteriors, m_contexts):
            pass

    def encode(
        self,
        inputs: List[Optional[torch.Tensor]],
        num_samples: int = None,
    ) -> Dict[Any, Any]:
        """Encode into modality-specific and -invariant latent space,
        with a hierarchical inference network

        Parameters
        ----------
        inputs : List[Optional[torch.Tensor]]
        num_samples : int, optional

        Returns
        -------
        Dict[Any, Any]
            {"m": m_latents,               "s": s_latent}
            {"m": List[Optional[B, Z]],    "s": [B, Z]}
            {"m": List[Optional[B, K, Z]], "s": [B, K, Z]}
        """
        # Encode into posterior dist parameters
        posterior_context = self.inputs_encoder(inputs)
        m_contexts = posterior_context["m"]
        s_context = posterior_context["s"]

        if num_samples is None:
            # Sample from s_posterior
            s_latent = self.s_posterior.sample(num_samples=1, context=s_context)
            s_latent = torchutils.merge_leading_dims(s_latent, num_dims=2)

            # Sample from m_posteriors, additionally conditioned on s_latent
            # Account for missing modalities
            m_latents = [
                None
                if context is None
                else torchutils.merge_leading_dims(
                    posterior.sample(
                        num_samples=1, context=torch.cat([context, s_latent])
                    ),
                    num_dims=2,
                )
                for posterior, context in zip(self.m_posteriors, m_contexts)
            ]

        else:
            # Sample from s_posterior
            s_latent = self.s_posterior.sample(
                num_samples=num_samples, context=s_context
            )  # [B, K, Z]

            # Sample from m_posteriors, additionally conditioned on s_latent
            m_latents = []

            for posterior, context in zip(self.m_posteriors, m_contexts):
                if context is None:
                    m_latents.append(None)

                else:
                    cat_context = torch.cat([
                        context.unsqueeze(1).repeat(1, num_samples, 1),
                        s_latent
                    ], dim=-1)

                    m_latents.append(
                        posterior.sample(num_samples=num_samples, context=cat_context)
                    )

            m_latents = [
                None
                if context is None
                else posterior.sample(
                    num_samples=num_samples,
                    context=torch.cat([context, s_latent], dim=-1),
                )
                for posterior, context in zip(self.m_posteriors, m_contexts)
            ]

        return {"m": m_latents, "s": s_latent}

    def decode(self, latents: Dict[Any, Any], mean: bool) -> List[torch.Tensor]:
        samples_list = []
        m_latents = latents["m"]
        s_latent = latents["s"]
        batch_size = s_latent.shape[0]

        # Get samples from each decoder
        for likelihood, prior, m_latent in zip(
            self.likelihoods, self.m_priors, m_latents
        ):
            # If missing m_latent, sample from prior instead
            if m_latent is None:
                # Modality-specific prior is conditioned on s_latent
                m_latent = prior.sample(1, context=s_latent)
                m_latent = torchutils.merge_leading_dims(m_latent, num_dims=2)

            # Concat modality-specific and -invariant latents
            concat_latent = torch.cat([m_latent, s_latent], dim=-1)

            if mean:
                samples = likelihood.mean(context=concat_latent)
            else:
                samples = likelihood.sample(num_samples=1, context=concat_latent)
                samples = torchutils.merge_leading_dims(samples, num_dims=2)

            samples_list.append(samples)

        return samples_list

    def sample(self, num_samples: int, mean=False) -> List[torch.Tensor]:
        s_latent = self.s_prior.sample(num_samples)
        # Modality-specific priors are conditioned on s_latent
        # Take note of tensor shapes
        m_latents = [
            torchutils.merge_leading_dims(prior.sample(1, context=s_latent), num_dims=2)
            for prior in self.m_priors
        ]

        return self.decode({"m": m_latents, "s": s_latent}, mean)


class HierPMVAE_v2(PartitionedMultimodalVAE):
    def decode(self, latents: Dict[Any, Any], mean: bool) -> List[torch.Tensor]:
        samples_list = []
        m_latents = latents["m"]
        s_latent = latents["s"]
        batch_size = s_latent.shape[0]

        # Get samples from each decoder
        for likelihood, prior, m_latent in zip(
            self.likelihoods, self.m_priors, m_latents
        ):
            # If missing m_latent, sample from prior instead
            if m_latent is None:
                # Modality-specific prior is conditioned on s_latent
                m_latent = prior.sample(1, context=s_latent)
                m_latent = torchutils.merge_leading_dims(m_latent, num_dims=2)

            # Don't concat latents; only condition on modality-specific latent
            if mean:
                samples = likelihood.mean(context=m_latent)
            else:
                samples = likelihood.sample(num_samples=1, context=m_latent)
                samples = torchutils.merge_leading_dims(samples, num_dims=2)

            samples_list.append(samples)

        return samples_list

    def sample(self, num_samples: int, mean=False) -> List[torch.Tensor]:
        s_latent = self.s_prior.sample(num_samples)
        # Modality-specific priors are conditioned on s_latent
        # Take note of tensor shapes
        m_latents = [
            torchutils.merge_leading_dims(prior.sample(1, context=s_latent), num_dims=2)
            for prior in self.m_priors
        ]

        return self.decode({"m": m_latents, "s": s_latent}, mean)
