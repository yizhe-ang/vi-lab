from src.utils import ConfigManager
import src.objectives as objectives
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.core.lightning import LightningModule
from src.callbacks import LatentDimInterpolator, VAEImageSampler
from src.models.vaes import VAE
from src.objectives import log_prob_lower_bound
from src.utils import ConfigManager


class VAE_Experiment(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # Create `hparams` attribute
        self.save_hyperparameters(hparams)
        self.config = ConfigManager(hparams)

        # Initialize datamodule
        self._init_datamodule()
        # Infer dataset shapes
        self.data_dim = self.datamodule.size()
        # Initialize callbacks
        self._init_callbacks()

        # Set-up nn modules according to `hparams`
        self._init_system()
        # Init objective function
        self.obj = getattr(objectives, hparams["objective"])

        # Maximum number of training steps
        self.max_steps = (
            len(self.datamodule.train_dataloader()) * self.hparams["max_epochs"]
        )

        # Have to rework forward calls for this to work
        # self.example_input_array = torch.randn(32, 1, 28, 28)

    def on_pretrain_routine_start(self):
        # Get number of parameters in model
        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger = self.logger.experiment
        logger.log({"n_parameters": n_parameters}, commit=False)

    def _init_datamodule(self):
        self.datamodule = self.config.init_object("datamodule")
        self.datamodule.prepare_data()
        self.datamodule.setup()

    def _init_system(self):
        """Set-up nn modules according to `hparams`"""
        latent_dim = self.hparams["latent_dim"]

        prior = self.config.init_object("prior", latent_dim)
        approx_posterior = self.config.init_object("approx_posterior", latent_dim)

        inputs_encoder = self._get_inputs_encoder()

        likelihood = self._get_likelihood()

        self._init_model(prior, approx_posterior, likelihood, inputs_encoder)

    def _init_model(self, prior, approx_posterior, likelihood, inputs_encoder):
        self.model = VAE(
            prior=prior,
            approximate_posterior=approx_posterior,
            likelihood=likelihood,
            inputs_encoder=inputs_encoder,
        )

    def _get_likelihood(self):
        return self.config.init_object("likelihood", self.hparams["latent_dim"])

    def _get_inputs_encoder(self):
        return self.config.init_object("inputs_encoder", self.hparams["latent_dim"] * 2)

    def _init_callbacks(self):
        self.callbacks = [
            VAEImageSampler(num_samples=64),
            LatentDimInterpolator(),
            LearningRateMonitor(logging_interval="step"),
        ]

    def _kl_multiplier(self):
        kl_warmup_fraction = self.hparams["kl_warmup_fraction"]
        kl_multiplier_initial = self.hparams["kl_multiplier_initial"]
        kl_multiplier_max = self.hparams["kl_multiplier_max"]

        multiplier = min(self.global_step / (self.max_steps * kl_warmup_fraction), 1.0,)

        return (
            kl_multiplier_initial
            + (kl_multiplier_max - kl_multiplier_initial) * multiplier
        )

    def _run_step(self, batch):
        x, _ = batch
        elbo = self.obj(self.model, x, kl_multiplier=self._kl_multiplier())

        return elbo.mean()

    def training_step(self, batch, batch_idx):
        elbo = self._run_step(batch)
        loss = -elbo

        self.log_dict(
            {"train_loss": loss, "kl_multiplier": torch.tensor(self._kl_multiplier())}
        )

        return loss

    def validation_step(self, batch, batch_idx):
        elbo = self._run_step(batch)

        self.log_dict({"val_elbo": elbo})

    def test_step(self, batch, batch_idx):
        elbo = self._run_step(batch)
        log_prob = log_prob_lower_bound(self.model, batch[0], num_samples=1000).mean()

        self.log_dict({"test_elbo": elbo, "test_log_prob": log_prob})

    def configure_optimizers(self):
        # scheduler = {
        #     "scheduler": optim.lr_scheduler.CosineAnnealingLR(
        #         optimizer, T_max=self.hparams["max_steps"], eta_min=0
        #     ),
        #     "interval": "step",
        # }

        # return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return self.config.init_object("optimizer", self.model.parameters())


class VAE_LangevinExperiment(VAE_Experiment):
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

        elbo, latents = self.obj(self.model, x, cached_latents, kl_multiplier=1.0)

        # Cache new samples
        self.cached_latents[indices] = latents.detach().clone()

        return elbo.mean()
