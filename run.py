import argparse

import pytorch_lightning as pl
import yaml
from pytorch_lightning.loggers import WandbLogger

import src.experiments as experiments
from src.callbacks import VAEImageSampler, LatentDimInterpolator


def main(hparams):
    pl.seed_everything(hparams["seed"])

    # Init logger
    wandb_logger = WandbLogger(name=hparams["name"], project="vae-expts",)
    wandb_logger.log_hyperparams(hparams)

    # Init experiment
    exp = getattr(experiments, hparams["experiment"])(hparams)

    # FIXME shift this to experiment object too?
    # Init callbacks
    callbacks = [VAEImageSampler(num_samples=8), LatentDimInterpolator()]

    # Init trainer
    trainer = pl.Trainer(
        deterministic=True,
        benchmark=True,
        callbacks=callbacks,
        early_stop_callback=False,
        # fast_dev_run=True,
        gpus=1,
        logger=wandb_logger,
        # reload_dataloaders_every_epoch=False,
        weights_summary="full",
        max_steps=hparams['max_steps'],
        limit_val_batches=0.,
        # gradient_clip_val=0.1
    )
    trainer.fit(exp, exp.datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE Experiment")
    parser.add_argument(
        "--config", "-c", help="path to the config file", default="configs/vae_flow.yaml"
    )

    args = parser.parse_args()

    # Load config file
    with open(args.config, "r") as file:
        try:
            hparams = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    main(hparams)
