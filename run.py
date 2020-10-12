import argparse
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import src.experiments as experiments
from src.experiments import MVAE_Tester


def main(hparams, resume):
    pl.seed_everything(hparams["seed"])

    # Init logger
    project_name = "vae-expts"
    wandb_logger = WandbLogger(
        name=hparams["name"], project=project_name, id=hparams["name"],  # For resuming
    )
    wandb_logger.log_hyperparams(hparams)

    # Init experiment
    exp = getattr(experiments, hparams["experiment"])(hparams)

    # ModelCheckpoint and EarlyStopping callbacks
    model_checkpoint = ModelCheckpoint(mode="max", save_last=True,)
    early_stop = EarlyStopping(
        patience=hparams["earlystop_patience"], mode="max", verbose=True
    )

    # Init trainer
    checkpoint_dir = Path("checkpoints")
    checkpoint_path = (
        checkpoint_dir / project_name / hparams["name"] / "checkpoints" / "last.ckpt"
    )

    trainer = pl.Trainer(
        # fast_dev_run=True,
        default_root_dir=checkpoint_dir,
        resume_from_checkpoint=str(checkpoint_path) if resume else None,
        deterministic=True,
        benchmark=True,
        callbacks=exp.callbacks,
        checkpoint_callback=model_checkpoint,
        # Disabling early stopping
        early_stop_callback=False,
        gpus=1,
        logger=wandb_logger,
        weights_summary="top",
        max_epochs=hparams["max_epochs"],
        min_epochs=hparams["min_epochs"],
        # val_check_interval=0.25,
        # auto_lr_find=True,
        # limit_val_batches=0.,
    )

    trainer.fit(exp, exp.datamodule)
    trainer.test(datamodule=exp.datamodule)

    mvae_tester = MVAE_Tester(exp)
    mvae_tester.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE Experiment")
    parser.add_argument(
        "--config", "-c", help="path to the config file", default="configs/config.yaml"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )

    args = parser.parse_args()

    # Load config file
    with open(args.config, "r") as file:
        try:
            hparams = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    main(hparams, resume=args.resume)
