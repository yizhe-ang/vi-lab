import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import model_checkpoint
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import src.experiments as experiments


def main(hparams):
    # HACK
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    pl.seed_everything(hparams["seed"])

    # torch.backends.cudnn.benchmark=False

    # Init logger
    wandb_logger = WandbLogger(name=hparams["name"], project="vae-expts",)
    wandb_logger.log_hyperparams(hparams)

    # Init experiment
    exp = getattr(experiments, hparams["experiment"])(hparams)

    model_checkpoint = ModelCheckpoint(
        monitor='val_elbo',
        mode='max'
    )

    # Init trainer
    trainer = pl.Trainer(
        deterministic=True,
        benchmark=True,
        callbacks=exp.callbacks,
        early_stop_callback=False,
        checkpoint_callback=model_checkpoint,
        # fast_dev_run=True,
        gpus=1,
        logger=wandb_logger,
        # reload_dataloaders_every_epoch=False,
        weights_summary="top",
        max_epochs=10_000,
        max_steps=hparams['max_steps']
        # limit_val_batches=0.,
        # gradient_clip_val=0.1
    )

    trainer.fit(exp, exp.datamodule)
    trainer.test(datamodule=exp.datamodule)


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
