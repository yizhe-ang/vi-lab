import numpy as np
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from src.callbacks import (
    CoherenceEvaluator,
    MultimodalVAE_ImageSampler,
    OnlineLinearProbe,
)
from src.models import (
    PartitionedMultimodalEncoder,
    PartitionedMultimodalVAE,
    HierPMVAE_v1,
    HierPMVAE_v2
)
from src.objectives import pmvae_elbo, hier_pmvae_elbo, hier_pmvae_v2_elbo

from . import VAE_Experiment


class PMVAE_Experiment(VAE_Experiment):
    def __init__(self, hparams):
        super().__init__(hparams)

        # HACK Only for bimodal
        self.likelihood_weights = (
            np.prod(self.data_dim[1]) / np.prod(self.data_dim[0]),
            1.0,
        )

    def _init_system(self):
        m_latent_dim = self.hparams["m_latent_dim"]
        s_latent_dim = self.hparams["s_latent_dim"]

        s_posterior = self.config.init_object("s_posterior", s_latent_dim)
        m_posteriors = self.config.init_objects("m_posteriors", m_latent_dim)

        s_prior = self.config.init_object("s_prior", s_latent_dim)
        m_priors = self.config.init_objects("m_priors", m_latent_dim)

        encoders = self.config.init_objects("encoders", m_latent_dim, s_latent_dim)
        fusion_module = self.config.init_object(
            "fusion_module",
            input_size=s_latent_dim * 2,
            output_size=s_latent_dim * 2,
            hidden_units=[s_latent_dim * 2],
        )
        inputs_encoder = PartitionedMultimodalEncoder(encoders, fusion_module)

        shapes = self.data_dim
        decoders = self.config.init_objects("decoders", m_latent_dim + s_latent_dim)
        likelihoods = self.config.init_objects("likelihoods", shapes, decoders)

        self.model = PartitionedMultimodalVAE(
            s_prior, m_priors, s_posterior, m_posteriors, likelihoods, inputs_encoder
        )

    def _run_step(self, batch):
        elbo = self.obj(
            self.model,
            batch["data"],
            self.likelihood_weights,
            kl_multiplier=self._kl_multiplier(),
        )

        return elbo.mean()

    def test_step(self, batch, batch_idx):
        num_samples = 1000

        # Get joint log prob (using importance sampling)
        elbo = pmvae_elbo(
            self.model, batch["data"], num_samples=num_samples, keepdim=True
        )
        log_prob = (
            torch.logsumexp(elbo, dim=1)
            - torch.log(torch.Tensor([num_samples]).to(self.device))
        ).mean()

        self.log_dict(
            {
                # "test_elbo": elbo,
                "test_log_prob": log_prob,
                # "test_acc": acc
            }
        )

    def _init_callbacks(self):
        self.callbacks = [
            MultimodalVAE_ImageSampler(include_modality=[True, True]),
            # MultimodalVAEReconstructor(self.datamodule.val_set),
            LearningRateMonitor(logging_interval="step"),
            OnlineLinearProbe(partitioned=True),
            CoherenceEvaluator(),
        ]


class HierPMVAE_v1_Experiment(PMVAE_Experiment):
    def _init_system(self):
        m_latent_dim = self.hparams["m_latent_dim"]
        s_latent_dim = self.hparams["s_latent_dim"]

        s_posterior = self.config.init_object("s_posterior", s_latent_dim)
        m_posteriors = self.config.init_objects("m_posteriors", m_latent_dim)

        s_prior = self.config.init_object("s_prior", s_latent_dim)
        m_priors = self.config.init_objects("m_priors", m_latent_dim, s_latent_dim)

        encoders = self.config.init_objects("encoders", m_latent_dim, s_latent_dim)
        fusion_module = self.config.init_object(
            "fusion_module",
            input_size=s_latent_dim * 2,
            output_size=s_latent_dim * 2,
            hidden_units=[s_latent_dim * 2],
        )
        inputs_encoder = PartitionedMultimodalEncoder(encoders, fusion_module)

        shapes = self.data_dim
        decoders = self.config.init_objects("decoders", m_latent_dim + s_latent_dim)
        likelihoods = self.config.init_objects("likelihoods", shapes, decoders)

        self.model = HierPMVAE_v1(
            s_prior, m_priors, s_posterior, m_posteriors, likelihoods, inputs_encoder
        )

    def test_step(self, batch, batch_idx):
        num_samples = 1000

        # Get joint log prob (using importance sampling)
        elbo = hier_pmvae_elbo(
            self.model, batch["data"], num_samples=num_samples, keepdim=True
        )
        log_prob = (
            torch.logsumexp(elbo, dim=1)
            - torch.log(torch.Tensor([num_samples]).to(self.device))
        ).mean()

        self.log_dict(
            {
                "test_log_prob": log_prob,
            }
        )

class HierPMVAE_v2_Experiment(PMVAE_Experiment):
    def _init_system(self):
        m_latent_dim = self.hparams["m_latent_dim"]
        s_latent_dim = self.hparams["s_latent_dim"]

        s_posterior = self.config.init_object("s_posterior", s_latent_dim)
        m_posteriors = self.config.init_objects("m_posteriors", m_latent_dim)

        s_prior = self.config.init_object("s_prior", s_latent_dim)
        m_priors = self.config.init_objects("m_priors", m_latent_dim, s_latent_dim)

        encoders = self.config.init_objects("encoders", m_latent_dim, s_latent_dim)
        fusion_module = self.config.init_object(
            "fusion_module",
            input_size=s_latent_dim * 2,
            output_size=s_latent_dim * 2,
            hidden_units=[s_latent_dim * 2],
        )
        inputs_encoder = PartitionedMultimodalEncoder(encoders, fusion_module)

        shapes = self.data_dim
        # Only conditioned on m_latent
        decoders = self.config.init_objects("decoders", m_latent_dim)
        likelihoods = self.config.init_objects("likelihoods", shapes, decoders)

        self.model = HierPMVAE_v2(
            s_prior, m_priors, s_posterior, m_posteriors, likelihoods, inputs_encoder
        )

    def test_step(self, batch, batch_idx):
        num_samples = 1000

        # Get joint log prob (using importance sampling)
        elbo = hier_pmvae_v2_elbo(
            self.model, batch["data"], num_samples=num_samples, keepdim=True
        )
        log_prob = (
            torch.logsumexp(elbo, dim=1)
            - torch.log(torch.Tensor([num_samples]).to(self.device))
        ).mean()

        self.log_dict(
            {
                "test_log_prob": log_prob,
            }
        )
