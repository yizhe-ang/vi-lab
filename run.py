import argparse

import pl_bolts.datamodules as datamodules
import pytorch_lightning as pl
import yaml
from pytorch_lightning.loggers import WandbLogger

import src.experiments as experiments

# from pl_bolts.callbacks import LatentDimInterpolator, TensorboardGenerativeModelImageSampler


def main(hparams):
    pl.seed_everything(hparams["seed"])

    # Init logger
    wandb_logger = WandbLogger(name=hparams["name"], project="vae-expts",)
    wandb_logger.log_hyperparams(hparams)

    # Init datamodule
    datamodule = getattr(datamodules, hparams["datamodule"])(
        **hparams["datamodule_args"]
    )

    # Init callbacks
    callbacks = None

    # Init experiment
    exp = getattr(experiments, hparams["experiment"])(hparams)

    # Init trainer
    trainer = pl.Trainer(
        deterministic=True,
        benchmark=True,
        callbacks=callbacks,
        early_stop_callback=False,
        fast_dev_run=True,
        gpus=1,
        logger=wandb_logger,
        reload_dataloaders_every_epoch=False,
        weights_summary="full",
        max_epochs=hparams["max_epochs"],
        datamodule=datamodule,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE Experiment")
    parser.add_argument(
        "--config", "-c", help="path to the config file", default="configs/config.yaml"
    )

    args = parser.parse_args()

    # Load config file
    with open(args.config, "r") as file:
        try:
            hparams = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    main(hparams)
