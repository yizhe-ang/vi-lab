import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim
from pytorch_lightning.metrics.functional import accuracy
from src.models.blocks import MLP
from torch.nn import functional as F


class OnlineLinearProbe(pl.Callback):
    def __init__(self):
        """
        Attaches a MLP for finetuning as per the standard self-supervised protocol,
        for MVAE

        pl_module should have:
            pl_module.hparams['latent_dim']
            pl_module.datamodule.n_classes
        """
        super().__init__()
        # To be set
        self.optimizer = None
        self.latent_dim = None
        self.n_classes = None  # Number of classes for each modality

    def on_pretrain_routine_start(self, trainer, pl_module):
        self.latent_dim = pl_module.hparams["latent_dim"]
        self.n_classes = pl_module.datamodule.n_classes
        n_modalities = len(pl_module.datamodule.dims)

        # Create linear probes for each modality, and joint modalities
        linear_probes = [
            MLP(self.latent_dim, self.n_classes)
            for _ in range(n_modalities + 1)
        ]
        pl_module.linear_probes = nn.ModuleList(linear_probes).to(pl_module.device)

        # Init optimizer
        self.optimizer = optim.Adam(pl_module.linear_probes.parameters(), lr=1e-3)

    def get_representations(self, pl_module, x1, x2):
        model = pl_module.model

        # Get latent representations from MVAE,
        # conditioned on all combination of modalities
        return [model.encode(data) for data in [[x1, None], [None, x2], [x1, x2]]]

    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        device = pl_module.device

        # Get data
        x1, x2 = [x.to(device) for x in batch["data"]]
        labels = batch['label'].to(device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x1, x2)

        # Forward pass through all respective linear probes
        preds = [probe(z) for z, probe in zip(representations, pl_module.linear_probes)]

        # FIXME Any way to vectorize this?
        losses = [F.cross_entropy(p, labels) for p in preds]
        # FIXME Is simply summing correct here?
        loss = torch.stack(losses).sum()

        # Update weights
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Log metrics
        logger = pl_module.logger.experiment

        accs = [accuracy(p, labels) for p in preds]
        metrics = {
            "train_m_probe": accs[0],
            "train_s_probe": accs[1],
            "train_m_s_probe": accs[2],
        }
        logger.log(metrics, commit=False)
