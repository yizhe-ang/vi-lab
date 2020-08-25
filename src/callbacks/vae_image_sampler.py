from pytorch_lightning import Callback
import torchvision
import wandb


class VAEImageSampler(Callback):
    def __init__(self, num_samples=3) -> None:
        """Generates images and logs to wandb

        Requirements:
            `pl_module` should have `img_dim` attribute
            `pl_module.vae` should have `sample` method implemented,
                z -> samples

        Parameters
        ----------
        num_samples : int, optional
            , by default 3
        """
        super().__init__()
        self.num_samples = num_samples

    def on_epoch_end(self, trainer, pl_module):

        # VAE model
        vae = pl_module.vae
        # Get samples
        images = vae.sample(self.num_samples, pl_module.device)

        # Create grid
        images = images.view(self.num_samples, *pl_module.img_dim)
        grid = torchvision.utils.make_grid(images)
        grid = grid.permute(1, 2, 0)

        # Log samples
        trainer.logger.experiment.log(
            {"samples": wandb.Image(grid)}, step=trainer.global_step
        )
