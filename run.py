import argparse

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import src.experiments as experiments


def main(hparams):
    pl.seed_everything(hparams["seed"])

    # Init logger
    wandb_logger = WandbLogger(name=hparams["name"], project="vae-expts",)
    wandb_logger.log_hyperparams(hparams)

    # Init experiment
    exp = getattr(experiments, hparams["experiment"])(hparams)

    # ModelCheckpoint and EarlyStopping callbacks
    model_checkpoint = ModelCheckpoint(
        mode='max',
    )
    early_stop = EarlyStopping(
        patience=5,
        mode='max',
    )

    # Init trainer
    trainer = pl.Trainer(
        # fast_dev_run=True,
        deterministic=True,
        benchmark=True,
        callbacks=exp.callbacks,
        checkpoint_callback=model_checkpoint,
        early_stop_callback=early_stop,
        gpus=1,
        logger=wandb_logger,
        weights_summary="top",
        max_epochs=10_000,
        max_steps=hparams['max_steps'],
        limit_val_batches=0.,
        # gradient_clip_val=0.1
    )

    trainer.fit(exp, exp.datamodule)
    # trainer.test(datamodule=exp.datamodule)


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
