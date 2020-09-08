import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.core.lightning import LightningModule
from torch import optim

import src.datamodules as datamodules
import src.models.dists as dists
import src.models.nns as nns
import src.objectives as objectives
from src.callbacks import LatentDimInterpolator, VAEImageSampler
from src.models.vaes import VariationalAutoencoder
from src.objectives import log_prob_lower_bound


class VAEExperiment(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # Create `hparams` attribute
        self.save_hyperparameters(hparams)
        # Initialize datamodule
        self._init_datamodule()
        # Initialize callbacks
        self._init_callbacks()
        # Set-up nn modules according to `hparams`
        self._init_system()
        # Init objective function
        self.obj = getattr(objectives, hparams["objective"])
        # Infer img dims
        self.img_dim = self.datamodule.size()

        # Have to rework forward calls for this to work
        # self.example_input_array = torch.randn(32, 1, 28, 28)

    def _init_system(self):
        """Set-up nn modules according to `hparams`"""
        latent_dim = self.hparams["latent_dim"]

        prior = getattr(dists, self.hparams["prior"])(
            latent_dim, **self.hparams["prior_args"]
        )
        approx_posterior = getattr(dists, self.hparams["approx_posterior"])(
            latent_dim, **self.hparams["approx_posterior_args"]
        )
        likelihood = getattr(dists, self.hparams["likelihood"])(
            latent_dim, **self.hparams["likelihood_args"]
        )

        if self.hparams["inputs_encoder"]:
            inputs_encoder = getattr(nns, self.hparams["inputs_encoder"])(
                latent_dim * 2, **self.hparams["inputs_encoder_args"]
            )
        else:
            inputs_encoder = None

        self.model = VariationalAutoencoder(
            prior=prior,
            approximate_posterior=approx_posterior,
            likelihood=likelihood,
            inputs_encoder=inputs_encoder,
        )

    def _init_datamodule(self):
        self.datamodule = getattr(datamodules, self.hparams["datamodule"])(
            **self.hparams["datamodule_args"]
        )

    def _init_callbacks(self):
        self.callbacks = [
            VAEImageSampler(num_samples=64),
            LatentDimInterpolator(),
            LearningRateLogger(logging_interval="step"),
        ]

    def _kl_multiplier(self):
        multiplier = min(
            self.global_step
            / (self.hparams["max_steps"] * self.hparams["kl_warmup_fraction"]),
            1.0,
        )

        return self.hparams["kl_multiplier_initial"] * (1.0 + multiplier)

    def _run_step(self, batch):
        x, _ = batch
        elbo = self.obj(self.model, x, kl_multiplier=self._kl_multiplier())

        return elbo.mean()

    def training_step(self, batch, batch_idx):
        elbo = self._run_step(batch)
        loss = -elbo

        result = pl.TrainResult(loss)
        result.log_dict({"train_loss": loss})

        return result

    def validation_step(self, batch, batch_idx):
        elbo = self._run_step(batch)

        result = pl.EvalResult(checkpoint_on=elbo, early_stop_on=elbo)
        result.log_dict({"val_elbo": elbo})

        return result

    def test_step(self, batch, batch_idx):
        elbo = self._run_step(batch)
        log_prob = log_prob_lower_bound(self.model, batch[0], num_samples=1000).mean()

        result = pl.EvalResult()
        result.log_dict({"test_elbo": elbo, "test_log_prob": log_prob})

        return result

    def configure_optimizers(self):
        optimizer = getattr(optim, self.hparams["optimizer"])(
            self.model.parameters(), **self.hparams["optimizer_args"]
        )
        scheduler = {
            "scheduler": optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams["max_steps"], eta_min=0
            ),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class VAELangevinExperiment(VAEExperiment):
    def __init__(self, hparams):
        super().__init__(hparams)

        # HACK
        self.cached_latents = torch.randn(
            self.datamodule.train_dataset_size, hparams["latent_dim"]
        ).cuda()

    def _run_step(self, batch):
        x, _, indices = batch

        # Get cached samples
        cached_latents = self.cached_latents[indices]

        elbo, latents = self.obj(
            self.model, x, cached_latents, kl_multiplier=self._kl_multiplier()
        )

        # Cache new samples
        self.cached_latents[indices] = latents.detach().clone()

        return elbo.mean()
