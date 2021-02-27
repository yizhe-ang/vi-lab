from pytorch_lightning.callbacks import LearningRateMonitor
from src.callbacks import (
    CoherenceEvaluator,
    MultimodalVAE_ImageSampler,
    OnlineLinearProbe,
)
from src.models import (
    HierPMVAE_v1,
    HierPMVAE_v2,
    PartitionedMultimodalEncoder,
    PartitionedMultimodalVAE,
)

from .mvae_expt import MVAE_Experiment


class PMVAE_Experiment(MVAE_Experiment):
    def __init__(self, hparams):
        super().__init__(hparams)

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
        # m_context: m_latent_dim * 2
        # s_latent: s_latent_dim
        m_posteriors = self.config.init_objects(
            "m_posteriors", m_latent_dim, m_latent_dim * 2 + s_latent_dim
        )

        s_prior = self.config.init_object("s_prior", s_latent_dim)
        m_priors = self.config.init_objects("m_priors", m_latent_dim, s_latent_dim)

        encoders = self.config.init_objects("encoders", m_latent_dim, s_latent_dim)
        fusion_module = self.config.init_object(
            "fusion_module",
            # input_size=s_latent_dim * 2,
            # output_size=s_latent_dim * 2,
            # hidden_units=[s_latent_dim * 2],
        )
        inputs_encoder = PartitionedMultimodalEncoder(encoders, fusion_module)

        shapes = self.data_dim
        decoders = self.config.init_objects("decoders", m_latent_dim + s_latent_dim)
        likelihoods = self.config.init_objects("likelihoods", shapes, decoders)

        self.model = HierPMVAE_v1(
            s_prior, m_priors, s_posterior, m_posteriors, likelihoods, inputs_encoder
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
