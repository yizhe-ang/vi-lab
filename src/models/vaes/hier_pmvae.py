import torch
from typing import List, Dict, Any, Optional
from nflows.utils import torchutils
from .pmvae import PartitionedMultimodalVAE


class HierPMVAE_v1(PartitionedMultimodalVAE):
    def log_q_z_x(
        self,
        inputs: List[Optional[torch.Tensor]] = None,
        latent=None,
        context=None,
        num_samples=1,
    ):
        # If inputs not specified (and latent and context specified instead)
        if not inputs:
            return self._log_q_z_x(latent, context)

        # Compute posterior contexts / parameters
        q_context = self.inputs_encoder(inputs)
        m_contexts = q_context["m"]  # [B, Z_m]
        s_context = q_context["s"]  # [B, Z_s]

        # Compute s_posterior
        s_latent, log_q_z_s = self.s_posterior.sample_and_log_prob(
            num_samples, s_context
        )
        s_latent = torchutils.merge_leading_dims(s_latent, num_dims=2)  # [B*K, Z]
        log_q_z_s = torchutils.merge_leading_dims(log_q_z_s, num_dims=2)  # [B*K]

        # Compute m_posteriors
        m_latents = []
        # To cache updated contexts
        cat_m_contexts = []
        log_q_z_ms = torch.zeros_like(log_q_z_s)

        for posterior, context in zip(self.m_posteriors, m_contexts):
            # FIXME How to sample multiple latents hierarchically?

            # Account for missing modalities
            if context is None:
                m_latents.append(None)
                cat_m_contexts.append(None)

            else:
                # Additionally conditioned on s_latent (concatenate to m_context)
                cat_context = torch.cat(
                    [torchutils.repeat_rows(context, num_samples), s_latent], dim=-1
                )

                m_latent, log_q_z_m = posterior.sample_and_log_prob(
                    1, context=cat_context
                )
                m_latent = torchutils.merge_leading_dims(
                    m_latent, num_dims=2
                )  # [B*K, Z]
                log_q_z_m = torchutils.merge_leading_dims(
                    log_q_z_m, num_dims=2
                )  # [B*K]

                cat_m_contexts.append(cat_context)
                m_latents.append(m_latent)
                log_q_z_ms += log_q_z_m

        log_prob = log_q_z_s + log_q_z_ms

        # log_prob, sampled latents, posterior context / parameters
        return (
            log_prob,
            {"m": m_latents, "s": s_latent},
            {"m": cat_m_contexts, "s": s_context},
        )

    def log_p_z(self, latents):
        m_latents = latents["m"]
        s_latent = latents["s"]
        batch_size = s_latent.shape[0]

        log_p_z_ms = torch.zeros(batch_size, device=s_latent.device)

        for prior, latent in zip(self.m_priors, m_latents):
            # Account for missing modalities
            if latent is None:
                continue

            # m_priors are conditioned on s_latent
            log_p_z_ms += prior.log_prob(latent, context=s_latent)

        log_p_z_s = self.s_prior.log_prob(s_latent)

        return log_p_z_ms + log_p_z_s

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
                        num_samples=1, context=torch.cat([context, s_latent], dim=-1)
                    ),
                    num_dims=2,
                )
                for posterior, context in zip(self.m_posteriors, m_contexts)
            ]

        else:
            # FIXME How to sample multiple latents hierarchically?
            # Sample from s_posterior
            s_latent = self.s_posterior.sample(
                num_samples=num_samples, context=s_context
            )  # [B, K, Z]

            # Sample from m_posteriors, additionally conditioned on s_latent
            m_latents = []  # [B, K, Z]

            for posterior, context in zip(self.m_posteriors, m_contexts):
                if context is None:
                    m_latents.append(None)

                else:
                    # context: [B, Z_m]
                    # s_latent: [B, K, Z_s]
                    cat_context = torch.cat(
                        [
                            torchutils.repeat_rows(context, num_samples),
                            torchutils.merge_leading_dims(s_latent, num_dims=2),
                        ],
                        dim=-1,
                    )  # [B*K, Z]

                    latent = posterior.sample(
                        num_samples=1, context=cat_context
                    )  # [B*K, 1, Z]
                    latent = torchutils.merge_leading_dims(latent, num_dims=2)
                    latent = torchutils.split_leading_dim(
                        latent, shape=[-1, num_samples]
                    )

                    m_latents.append(latent)

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