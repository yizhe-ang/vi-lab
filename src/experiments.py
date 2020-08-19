from torch import optim
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl

import src.models.encoders as encoders
import src.models.decoders as decoders
import src.models.vaes as vaes
import src.losses as losses


class Experiment(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # Create `hparams` attribute
        self.save_hyperparameters(hparams)
        # Set-up nn modules according to `hparams`
        self._init_system()
        # Initialize loss function
        self.loss = getattr(losses, self.hparams["loss"])

    def _init_system(self):
        """Set-up nn modules according to `hparams`
        """
        encoder_args = self.hparams["encoder_args"] or {}
        encoder = getattr(encoders, self.hparams["encoder"])(**encoder_args)

        decoder_args = self.hparams["decoder_args"] or {}
        decoder = getattr(decoders, self.hparams["decoder"])(**decoder_args)

        vae_args = self.hparams["vae_args"] or {}
        self.vae = getattr(vaes, self.hparams["vae"])(encoder, decoder, **vae_args)

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
        result = pl.EvalResult()
        result.log_dict(
            {"val_loss": loss, "val_recon_loss": recon_loss, "val_kl_div": kl_div,}
        )

        return result

    def test_step(self, batch, batch_idx):
        loss, recon_loss, kl_div = self._run_step(batch)
        result = pl.EvalResult()
        result.log_dict(
            {"test_loss": loss, "test_recon_loss": recon_loss, "test_kl_div": kl_div,}
        )
        return result

    def configure_optimizers(self):
        optimizer_args = self.hparams["optimizer_args"] or {}

        return getattr(optim, self.hparams["optimizer"])(
            self.vae.parameters(), **optimizer_args
        )
