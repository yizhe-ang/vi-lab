import torch
import torchvision
import wandb
from pytorch_lightning import Callback
from torch.utils.data import Dataset
import numpy as np


class MultimodalVAEReconstructor(Callback):
    def __init__(self, dataset: Dataset, num_samples=8) -> None:
        """Generates cross-reconstructions and logs to wandb

        Requirements:
            `pl_module` should have `data_dim` attribute
            `pl_module.model` should have `sample` method implemented,
                z -> samples

        Parameters
        ----------
        dataset : Dataset
            Dataset to sample from
        num_samples : int, optional
            , by default 8
        """
        super().__init__()

        self.dataset = dataset

        # Choose random samples to reconstruct
        self.samples_idx = np.random.randint(len(dataset), size=num_samples)


    def on_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.eval()

            # VAE model
            model = pl_module.model
            # Get samples
            samples = model.sample(self.num_samples, mean=True)

            # Create grid for each modality
            for i, s in enumerate(samples):
                # FIXME Do I need this?
                # images = images.view(self.num_samples, *pl_module.data_dim)
                grid = torchvision.utils.make_grid(s, nrow=8)
                grid = grid.permute(1, 2, 0).cpu().numpy()

                # Log samples
                trainer.logger.experiment.log(
                    {f"samples_{i}": wandb.Image(grid)}, commit=False
                )

            # Cross reconstruction

        pl_module.train()
