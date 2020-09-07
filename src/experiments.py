from functools import partial

import pytorch_lightning as pl
import torch
from pytorch_lightning.core.lightning import LightningModule
from torch import optim

import src.datamodules as datamodules
import src.losses as losses
import src.models.decoders as decoders
import src.models.encoders as encoders
import src.models.vaes as vaes
from src.models.vaes import VAE
import src.nf as nf


class VAEExperiment(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # Create `hparams` attribute
        self.save_hyperparameters(hparams)
        # Initialize datamodule
        self._init_datamodule()
        # Set-up nn modules according to `hparams`
        self._init_system()
        # Infer img dims
        self.img_dim = self.datamodule.size()
        # Initialize loss function
        self.loss = getattr(losses, self.hparams["loss"])
        loss_args = self.hparams["loss_args"]
        if loss_args:
            self.loss = partial(self.loss, **loss_args)

        # Have to rework forward calls for this to work
        # self.example_input_array = torch.randn(32, 1, 28, 28)

    def _init_system(self):
        """Set-up nn modules according to `hparams`
        """
        encoder = self._init_encoder()
        decoder = self._init_decoder()

        vae_args = self.hparams["vae_args"] or {}
        self.vae = getattr(vaes, self.hparams["vae"])(encoder, decoder, **vae_args)

    def _init_encoder(self):
        encoder_args = self.hparams["encoder_args"] or {}
        encoder = getattr(encoders, self.hparams["encoder"])(
            self.hparams["z_dim"], **encoder_args
        )

        return encoder

    def _init_decoder(self):
        decoder_args = self.hparams["decoder_args"] or {}
        decoder = getattr(decoders, self.hparams["decoder"])(
            self.hparams["z_dim"], **decoder_args
        )

        return decoder

    def _init_datamodule(self):
        self.datamodule = getattr(datamodules, self.hparams["datamodule"])(
            **self.hparams["datamodule_args"]
        )

    def _kl_multiplier(self):
        multiplier = min(self.global_step / (self.hparams["max_steps"] * 0.1), 1.0,)

        return 0.5 * (1.0 + multiplier)

    def _run_step(self, batch):
        _, x, _ = batch
        loss, recon_loss, kl_div = self.loss(
            self.vae, x, indices=None, kl_multiplier=self._kl_multiplier()
        )

        return loss, recon_loss, kl_div

    def training_step(self, batch, batch_idx):
        loss, recon_loss, kl_div = self._run_step(batch)

        result = pl.TrainResult(loss)
        result.log_dict(
            {"train_loss": loss, "train_recon_loss": recon_loss, "train_kl_div": kl_div}
        )

        return result

    def validation_step(self, batch, batch_idx):
        loss, recon_loss, kl_div = self._run_step(batch)
        result = pl.EvalResult(loss)
        result.log_dict(
            {"val_loss": loss, "val_recon_loss": recon_loss, "val_kl_div": kl_div,}
        )

        return result

    def test_step(self, batch, batch_idx):
        loss, recon_loss, kl_div = self._run_step(batch)
        result = pl.EvalResult(loss)
        result.log_dict(
            {"test_loss": loss, "test_recon_loss": recon_loss, "test_kl_div": kl_div,}
        )
        return result

    def configure_optimizers(self):
        optimizer_args = self.hparams["optimizer_args"] or {}

        return getattr(optim, self.hparams["optimizer"])(
            self.vae.parameters(), **optimizer_args
        )


class VAELangevinExperiment(VAEExperiment):
    def __init__(self, hparams):
        super().__init__(hparams)

    def _init_encoder(self):
        z_dim = self.hparams["z_dim"]
        dataset_size = self.datamodule.train_dataset_size
        device = self.device

        encoder_args = self.hparams["encoder_args"] or {}
        encoder = getattr(encoders, self.hparams["encoder"])(
            z_dim, dataset_size, device, **encoder_args
        )

        return encoder

    def _run_step(self, batch):
        indices, x, _ = batch
        loss, recon_loss, kl_div = self.loss(self.vae, x, indices)

        return loss, recon_loss, kl_div


class VAEFlowExperiment(VAEExperiment):
    def __init__(self, hparams):
        super().__init__(hparams)

    def _init_encoder(self):
        # VAE + Flow encoder
        z_dim = self.hparams["z_dim"]

        # Init flow
        flow_args = self.hparams["flow_args"] or {}
        flow = getattr(nf, self.hparams["flow"])(z_dim, **flow_args)

        encoder_args = self.hparams["encoder_args"] or {}
        encoder = getattr(encoders, self.hparams["encoder"])(
            z_dim, flow, **encoder_args
        )

        return encoder
