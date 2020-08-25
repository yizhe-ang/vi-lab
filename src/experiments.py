import pytorch_lightning as pl
import torch
from pytorch_lightning.core.lightning import LightningModule
from torch import optim

import src.datamodules as datamodules
import src.losses as losses
import src.models.decoders as decoders
import src.models.encoders as encoders
import src.models.vaes as vaes


class Experiment(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # Create `hparams` attribute
        self.save_hyperparameters(hparams)
        # Set-up nn modules according to `hparams`
        self._init_system()
        # Initialize datamodule
        self._init_datamodule()
        # Infer img dims
        self.img_dim = self.datamodule.size()
        # Initialize loss function
        self.loss = getattr(losses, self.hparams["loss"])

        # Have to rework forward calls for this to work
        # self.example_input_array = torch.randn(32, 1, 28, 28)

    def _init_system(self):
        """Set-up nn modules according to `hparams`
        """
        encoder_args = self.hparams["encoder_args"] or {}
        encoder = getattr(encoders, self.hparams["encoder"])(
            self.hparams["z_dim"], **encoder_args
        )

        decoder_args = self.hparams["decoder_args"] or {}
        decoder = getattr(decoders, self.hparams["decoder"])(
            self.hparams["z_dim"], **decoder_args
        )

        vae_args = self.hparams["vae_args"] or {}
        self.vae = getattr(vaes, self.hparams["vae"])(encoder, decoder, **vae_args)

    def _init_datamodule(self):
        self.datamodule = getattr(datamodules, self.hparams["datamodule"])(
            **self.hparams["datamodule_args"]
        )

    def _run_step(self, batch):
        x, _ = batch
        loss, recon_loss, kl_div = self.loss(self.vae, x)

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
