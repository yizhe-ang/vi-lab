import numpy as np
import pytorch_lightning as pl
import src.models.dists as dists
import src.models.nns as nns
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.metrics.functional import accuracy
from src.callbacks import MultimodalVAE_ImageSampler
from src.models.nns import MultimodalEncoder, ProductOfExpertsEncoder
from src.models.vaes import MultimodalVAE
from src.objectives import log_prob_lower_bound

from . import VAE_Experiment


class MVAE_Experiment(VAE_Experiment):
    def __init__(self, hparams):
        super().__init__(hparams)

        # HACK Only for bimodal
        self.likelihood_weights = (
            np.prod(self.data_dim[1]) / np.prod(self.data_dim[0]),
            1.0,
        )
        # self.likelihood_weights = self.datamodule.likelihood_weights

    def _init_model(self, prior, approx_posterior, likelihood, inputs_encoder):
        self.model = MultimodalVAE(prior, approx_posterior, likelihood, inputs_encoder)

    def _get_likelihood(self):
        # Get shapes of dataset
        shapes = self.data_dim
        latent_dim = self.hparams["latent_dim"]

        # Get decoders
        decoders = [
            getattr(nns, d)(latent_dim, **args)
            for d, args in zip(self.hparams["decoder"], self.hparams["decoder_args"])
        ]

        likelihood = self.hparams["likelihood"]
        likelihood_args = self.hparams["likelihood_args"]

        return [
            getattr(dists, l)(s, d, **args)
            for l, args, s, d in zip(likelihood, likelihood_args, shapes, decoders)
        ]

    def _get_inputs_encoder(self):
        raise NotImplementedError

    def _run_step(self, batch):
        elbo = self.obj(
            self.model,
            batch["data"],
            self.likelihood_weights,
            kl_multiplier=self._kl_multiplier(),
        )

        return elbo.mean()

    def test_step(self, batch, batch_idx):
        elbo = self._run_step(batch)
        # Get joint log prob (using importance sampling)
        log_prob = log_prob_lower_bound(
            self.model, batch["data"], num_samples=1000
        ).mean()
        # acc = self._classification_accuracy(batch["data"])

        result = pl.EvalResult()
        result.log_dict(
            {
                "test_elbo": elbo,
                "test_log_prob": log_prob,
                # "test_acc": acc
            }
        )

        return result

    # FIXME Helper function
    def _classification_accuracy(self, inputs):
        x, y = inputs
        x_recons, y_recons = self.model.cross_reconstruct(inputs, mean=True)

        return accuracy(y_recons, y)

    def _init_callbacks(self):
        self.callbacks = [
            MultimodalVAE_ImageSampler(include_modality=[True, True]),
            # MultimodalVAEReconstructor(self.datamodule.val_set),
            LearningRateLogger(logging_interval="step"),
        ]


class PoE_MVAE_Experiment(MVAE_Experiment):
    def _get_inputs_encoder(self):
        latent_dim = self.hparams["latent_dim"]

        encoders = [
            getattr(nns, e)(latent_dim, **args)
            for e, args in zip(self.hparams["encoder"], self.hparams["encoder_args"])
        ]

        return ProductOfExpertsEncoder(latent_dim, encoders)


class Fusion_MVAE_Experiment(MVAE_Experiment):
    def _get_inputs_encoder(self):
        latent_dim = self.hparams["latent_dim"]

        encoders = [
            getattr(nns, e)(latent_dim, **args)
            for e, args in zip(self.hparams["encoder"], self.hparams["encoder_args"])
        ]

        fusion_module = getattr(nns, self.hparams['fusion_module'])(
            input_size=latent_dim*2,
            output_size=latent_dim*2,
            hidden_units=[latent_dim*2],
        )

        return MultimodalEncoder(encoders, fusion_module)
