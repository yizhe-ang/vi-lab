import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.metrics.functional import accuracy
from src.models.base import MLP
from torch import optim
from torch.nn import functional as F


class OnlineLinearProbe(pl.Callback):
    def __init__(self, partitioned=False):
        """
        Attaches a MLP for finetuning as per the standard self-supervised protocol,
        for MVAE

        pl_module should have:
            pl_module.hparams['latent_dim']
            pl_module.datamodule.n_classes
        """
        super().__init__()
        self.partitioned = partitioned  # If latent space is partitioned
        # To be set
        self.optimizer = None
        self.n_classes = None  # Number of classes for each modality

    def _concat_latents(self, latents):
        # For partitioned latents
        m_latents = latents["m"]
        s_latent = latents["s"]

        return torch.cat([l for l in m_latents if l is not None] + [s_latent], dim=-1)

    def on_pretrain_routine_start(self, trainer, pl_module):
        # FIXME Attach to different (earlier) callback hook?
        self.n_classes = pl_module.datamodule.n_classes
        n_modalities = len(pl_module.datamodule.dims)

        if not self.partitioned:
            self.latent_dim = pl_module.hparams["latent_dim"]
            # Create linear probes for each modality, and joint modalities
            linear_probes = [
                MLP(self.latent_dim, self.n_classes) for _ in range(n_modalities + 1)
            ]

        else:
            self.m_latent_dim = pl_module.hparams["m_latent_dim"]
            self.s_latent_dim = pl_module.hparams["s_latent_dim"]

            # Create linear probes for each modality, and joint modalities
            # FIXME To include modality-specific latent?
            linear_probes = [
                MLP(self.m_latent_dim + self.s_latent_dim, self.n_classes),
                MLP(self.m_latent_dim + self.s_latent_dim, self.n_classes),
                MLP(self.m_latent_dim * 2 + self.s_latent_dim, self.n_classes),
            ]

        # FIXME Shouldn't attach to `pl_module`?
        pl_module.linear_probes = nn.ModuleList(linear_probes).to(pl_module.device)

        # Init optimizer
        self.optimizer = optim.Adam(pl_module.linear_probes.parameters(), lr=1e-3)

    def get_representations(self, pl_module, x1, x2):
        model = pl_module.model

        # Get latent representations from MVAE,
        # conditioned on all combination of modalities
        if not self.partitioned:
            return [model.encode(data) for data in [[x1, None], [None, x2], [x1, x2]]]
        else:
            return [
                self._concat_latents(model.encode(data))
                for data in [[x1, None], [None, x2], [x1, x2]]
            ]

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        device = pl_module.device

        # Get data
        x1, x2 = [x.to(device) for x in batch["data"]]
        labels = batch["label"].to(device)

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
        accs = [accuracy(p, labels) for p in preds]

        metrics = {
            "train_m_probe": accs[0],
            "train_s_probe": accs[1],
            "train_m_s_probe": accs[2],
        }
        # logger = pl_module.logger.experiment
        # logger.log(metrics, commit=False)

        # FIXME This doesn't work
        pl_module.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=False)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        device = pl_module.device

        # Get data
        x1, x2 = [x.to(device) for x in batch["data"]]
        labels = batch["label"].to(device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x1, x2)

            # Forward pass through all respective linear probes
            preds = [
                probe(z) for z, probe in zip(representations, pl_module.linear_probes)
            ]

        # Log metrics
        accs = [accuracy(p, labels) for p in preds]

        metrics = {
            "val_m_probe": accs[0],
            "val_s_probe": accs[1],
            "val_m_s_probe": accs[2],
        }

        pl_module.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        device = pl_module.device

        # Get data
        x1, x2 = [x.to(device) for x in batch["data"]]
        labels = batch["label"].to(device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x1, x2)

            # Forward pass through all respective linear probes
            preds = [
                probe(z) for z, probe in zip(representations, pl_module.linear_probes)
            ]

        # Log metrics
        accs = [accuracy(p, labels) for p in preds]

        metrics = {
            "test_m_probe": accs[0],
            "test_s_probe": accs[1],
            "test_m_s_probe": accs[2],
        }

        pl_module.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)
