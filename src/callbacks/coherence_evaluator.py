from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy
from src.models.classifiers import MNIST_Classifier, SVHN_Classifier


class CoherenceEvaluator(pl.Callback):
    def __init__(self):
        super().__init__()

    def _cross_coherence(self, pl_module, batch):
        # Compute Cross Coherence
        model = pl_module.model
        device = pl_module.device

        # Get data
        mnist, svhn = [x.to(device) for x in batch["data"]]
        targets = batch["label"].to(device)

        with torch.no_grad():
            # Get cross reconstructions
            m_recons, s_recons = model.cross_reconstruct([mnist, svhn], mean=True)

            # Get predictions
            m_preds = self.mnist_net(m_recons)
            s_preds = self.svhn_net(s_recons)

        corr_m = accuracy(m_preds, targets)
        corr_s = accuracy(s_preds, targets)

        return corr_m, corr_s

    def _joint_coherence(self, pl_module):
        # Compute Joint Coherence
        n_samples = 10_000
        corr = 0

        model = pl_module.model
        model.eval()

        with torch.no_grad():
            mnist, svhn = model.sample(n_samples, mean=True)

            # Get predictions
            m_preds = self.mnist_net(mnist).argmax(dim=1)
            s_preds = self.svhn_net(svhn).argmax(dim=1)

        # Evaluate correct samples
        corr += (m_preds == s_preds).sum().item()
        acc = corr / n_samples

        return acc

    def on_pretrain_routine_start(self, trainer, pl_module):
        # FIXME Or `on_test_start`

        # Init classifiers
        # Load pretrained weights
        mnist_weights_path = Path("saved") / "mnist_svhn" / "mnist_model.pt"
        svhn_weights_path = Path("saved") / "mnist_svhn" / "svhn_model.pt"

        self.mnist_net = MNIST_Classifier().to(pl_module.device)
        self.svhn_net = SVHN_Classifier().to(pl_module.device)

        self.mnist_net.load_state_dict(torch.load(mnist_weights_path))
        self.svhn_net.load_state_dict(torch.load(svhn_weights_path))

        self.mnist_net.eval()
        self.svhn_net.eval()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        corr_m, corr_s = self._cross_coherence(pl_module, batch)

        metrics = {
            "val_cross_coherence_s_m": corr_m,
            "val_cross_coherence_m_s": corr_s,
        }
        pl_module.log_dict(metrics, on_step=False, on_epoch=True)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        corr_m, corr_s = self._cross_coherence(pl_module, batch)

        metrics = {
            "test_cross_coherence_s_m": corr_m,
            "test_cross_coherence_m_s": corr_s,
        }
        pl_module.log_dict(metrics, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        acc = self._joint_coherence(pl_module)

        metrics = {"test_joint_coherence": acc}
        pl_module.log_dict(metrics, on_step=False, on_epoch=True)
