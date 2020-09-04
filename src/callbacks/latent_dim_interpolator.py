import torch
from pytorch_lightning.callbacks import Callback
import torchvision
import wandb


class LatentDimInterpolator(Callback):
    def __init__(
        self, interpolate_epoch_interval=20, range_start=-5, range_end=5
    ) -> None:
        """Interpolates the latent space for a model by setting all dims to zero
        and steping through the first two dims increasing one unit at a time,
        and logs to wandb

        Default interpolates between [-5, 5] (-5, -4, -3, ..., 3, 4, 5)

        Requirements:
            `pl_module` should have `img_dim` attribute

        Parameters
        ----------
        interpolate_epoch_interval : int, optional
            [description], by default 20
        range_start : int, optional
            [description], by default -5
        range_end : int, optional
            [description], by default 5
        """
        super().__init__()

        self.interpolate_epoch_interval = interpolate_epoch_interval
        self.range_start = range_start
        self.range_end = range_end

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.interpolate_epoch_interval == 0:
            images = self.interpolate_latent_space(pl_module)
            images = torch.cat(images, dim=0)

            # Create grid
            range_len = self.range_end - self.range_start
            grid = torchvision.utils.make_grid(images, nrow=range_len)
            grid = grid.permute(1, 2, 0).cpu().numpy()

            # Log samples
            trainer.logger.experiment.log(
                {"latent_interpolation": wandb.Image(grid)}, commit=False
            )

    def interpolate_latent_space(self, pl_module):
        images = []

        model = pl_module.model
        latent_dim = pl_module.hparams['latent_dim']

        with torch.no_grad():
            pl_module.eval()

            for z1 in range(self.range_start, self.range_end, 1):
                for z2 in range(self.range_start, self.range_end, 1):
                    # Set all dims to zero
                    z = torch.zeros(1, latent_dim, device=pl_module.device)

                    # Set the first 2 dims to the value
                    z[:, 0] = torch.tensor(z1)
                    z[:, 1] = torch.tensor(z2)

                    # Generate samples
                    img = model.decode(z, mean=True)
                    img = img.view(1, *pl_module.img_dim)

                    images.append(img)

        pl_module.train()
        return images
