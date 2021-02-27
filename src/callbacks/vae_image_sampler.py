import torch
import torchvision
import wandb
from pytorch_lightning import Callback


class VAEImageSampler(Callback):
    def __init__(self, num_samples=64) -> None:
        """Generates images and logs to wandb

        Requirements:
            `pl_module` should have `data_dim` attribute
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
        with torch.no_grad():
            pl_module.eval()

            # VAE model
            model = pl_module.model
            # Get samples
            images = model.sample(self.num_samples, mean=True)

            # Create grid

            # FIXME Do I need this?
            # images = images.view(self.num_samples, *pl_module.data_dim)
            grid = torchvision.utils.make_grid(images, nrow=8)
            grid = grid.permute(1, 2, 0).cpu().numpy()

            # Log samples
            trainer.logger.experiment.log(
                {"samples": wandb.Image(grid)}, commit=False
            )

        pl_module.train()
