import numpy as np
import src.objectives as objectives
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.core.lightning import LightningModule
from src.callbacks import (
    CoherenceEvaluator,
    MultimodalVAE_ImageSampler,
    OnlineLinearProbe,
    CelebaEvaluator,
    CelebaLinearProbe,
)
from src.models import MultimodalEncoder, ProductOfExpertsEncoder
from src.models.vaes import MultimodalVAE
from src.objectives import stochastic_elbo
from src.utils import ConfigManager


class MVAE_Experiment(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.nan_loss = False

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

        # HACK Only for bimodal
        # self.likelihood_weights = (
        #     np.prod(self.data_dim[1]) / np.prod(self.data_dim[0]),
        #     1.0,
        # )
        self.likelihood_weights = self.datamodule.likelihood_weights

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

        likelihoods = self._get_likelihoods()

        self._init_model(prior, approx_posterior, likelihoods, inputs_encoder)

    def _init_model(self, prior, approx_posterior, likelihood, inputs_encoder):
        self.model = MultimodalVAE(prior, approx_posterior, likelihood, inputs_encoder)

    def _get_likelihoods(self):
        # Get shapes of dataset
        shapes = self.data_dim
        latent_dim = self.hparams["latent_dim"]

        # Get decoders
        decoders = self.config.init_objects("decoder", latent_dim)

        # FIXME Double check how likelihoods are initialized
        return self.config.init_objects("likelihood", shapes, decoders)

    def _get_inputs_encoder(self):
        # raise NotImplementedError
        return None

    def _init_callbacks(self):
        self.callbacks = [
            MultimodalVAE_ImageSampler(include_modality=[True, True]),
            # MultimodalVAEReconstructor(self.datamodule.val_set),
            # LearningRateMonitor(logging_interval="step"),
            OnlineLinearProbe(),
            CoherenceEvaluator(),
        ]

    def _kl_multiplier(self):
        kl_warmup_fraction = self.hparams["kl_warmup_fraction"]
        kl_multiplier_initial = self.hparams["kl_multiplier_initial"]
        kl_multiplier_max = self.hparams["kl_multiplier_max"]

        multiplier = min(
            self.global_step / (self.max_steps * kl_warmup_fraction),
            1.0,
        )

        return (
            kl_multiplier_initial
            + (kl_multiplier_max - kl_multiplier_initial) * multiplier
        )

    def _run_step(self, batch):
        elbo = self.obj(
            self.model,
            batch,
            self.likelihood_weights,
            kl_multiplier=self._kl_multiplier(),
        )

        return elbo.mean()

    # def training_step(self, batch, batch_idx):
    #     opt = self.optimizers()

    #     def optimizer_closure():
    #         elbo = self._run_step(batch)
    #         loss = -elbo

    #         self.log("train_loss", loss)
    #         self.log("kl_multiplier", torch.tensor(self._kl_multiplier()))

    #         # Compute gradients
    #         self.manual_backward(loss, opt)

    #     # grad_norm = self._compute_grad_norm()
    #     # self.log("kl_multiplier", torch.tensor(self._kl_multiplier()))

    #     # print(grad_norm)

    #     # Gradient skipping
    #     # if not grad_norm >= self.hparams["gradient_skip_threshold"]:

    #     #     self.manual_optimizer_step(opt, optimizer_closure=optimizer_closure)
    #     #     # opt.step()
    #     # else:
    #     #     # Don't perform gradient update
    #     #     opt.zero_grad()

    #     self.manual_optimizer_step(opt, optimizer_closure=optimizer_closure)
    #     opt.zero_grad()

    def training_step(self, batch, batch_idx):
        elbo = self._run_step(batch)
        loss = -elbo
        # Check for nan loss
        self.nan_loss = torch.isnan(loss).item()

        self.log_dict(
            {
                "train_loss": loss,
                "kl_multiplier": torch.tensor(self._kl_multiplier()),
                "nan_loss": int(self.nan_loss),
            }
        )

        return loss

    def on_after_backward(self):
        gradient_clip = self.hparams["gradient_clip"]
        gradient_skip_thresh = self.hparams["gradient_skip_thresh"]

        # Clip gradient norm
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip).item()

        # Only update if no nan loss, and if grad norm is below a specific threshold
        if not self.nan_loss and (
            gradient_skip_thresh == -1 or grad_norm < gradient_skip_thresh
        ):
            skipped_update = 0
        else:
            skipped_update = 1
            self.model.zero_grad()

        self.nan_loss = False

        self.log("grad_norm", grad_norm, on_step=True, on_epoch=False)
        self.log("skipped_update", skipped_update, on_step=True, on_epoch=False)

    # def _compute_grad_norm(self):
    #     norm_type = 2.0

    #     parameters = [p for p in self.model.parameters() if p.grad is not None]
    #     # If no gradient information
    #     if not parameters:
    #         print("#######################################################")
    #         print("parameters is empty!!")
    #         total_norm = 0
    #     else:
    #         print("#######################################################")
    #         print("parameters is NOT empty!!")
    #         # total_norm = torch.norm(
    #         #     torch.stack(
    #         #         [torch.norm(p.grad.detach(), norm_type) for p in parameters]
    #         #     ),
    #         #     norm_type,
    #         # )
    #         print("#######################################################")
    #         print(parameters[1].grad)

    #         total_norm = torch.stack(
    #             [torch.norm(p.grad.detach(), norm_type) for p in parameters]
    #         )

    #     return total_norm

    # def optimizer_step(self, *args, **kwargs):
    #     # Compute grad_norm
    #     grad_norm = self._compute_grad_norm()
    #     # print("#######################################################")
    #     # print(grad_norm)
    #     # self.log("grad_norm", grad_norm)

    #     # Gradient skipping
    #     # if not grad_norm >= self.hparams["gradient_skip_threshold"]:
    #     super().optimizer_step(*args, **kwargs)

    def validation_step(self, batch, batch_idx):
        elbo = self._run_step(batch)

        self.log_dict({"val_elbo": elbo})

    def test_step(self, batch, batch_idx):
        num_samples = 1000

        # Get joint log prob (using importance sampling)
        elbo = stochastic_elbo(
            self.model, batch["data"], num_samples=num_samples, keepdim=True
        )
        log_prob = (
            torch.logsumexp(elbo, dim=1)
            - torch.log(torch.Tensor([num_samples]).to(self.device))
        ).mean()

        self.log_dict({"test_log_prob": log_prob})

    def configure_optimizers(self):
        # scheduler = {
        #     "scheduler": optim.lr_scheduler.CosineAnnealingLR(
        #         optimizer, T_max=self.hparams["max_steps"], eta_min=0
        #     ),
        #     "interval": "step",
        # }

        # return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return self.config.init_object("optimizer", self.model.parameters())


class PoE_MVAE_Experiment(MVAE_Experiment):
    def _get_inputs_encoder(self):
        latent_dim = self.hparams["latent_dim"]

        encoders = self.config.init_objects("encoder", latent_dim)

        return ProductOfExpertsEncoder(latent_dim, encoders)


class Fusion_MVAE_Experiment(MVAE_Experiment):
    def _get_inputs_encoder(self):
        latent_dim = self.hparams["latent_dim"]

        encoders = self.config.init_objects("encoder", latent_dim)
        fusion_module = self.config.init_object(
            "fusion_module",
        )

        return MultimodalEncoder(encoders, fusion_module)

class Celeba_MVAE_Experiment(Fusion_MVAE_Experiment):
    def _init_callbacks(self):
        self.callbacks = [
            MultimodalVAE_ImageSampler(include_modality=[True, False]),
            CelebaEvaluator(),
            CelebaLinearProbe(),
        ]
