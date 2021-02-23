from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.classification import AveragePrecision
from src.models.celeba import CelebaTextClassifier, CelebaImgClassifier
from src.utils import CELEBA_CLASSES


class CelebaEvaluator(pl.Callback):
    def __init__(self):
        super().__init__()

        # Compute average precision for each class
        self.aps = [(AveragePrecision(), AveragePrecision()) for _ in CELEBA_CLASSES]

    def on_pretrain_routine_start(self, trainer, pl_module):
        # FIXME Or `on_test_start`

        # Init classifiers
        # Load pretrained weights
        img_weights_path = Path("saved") / "celeba" / "clf_m1"
        text_weights_path = Path("saved") / "celeba" / "clf_m2"

        # FIXME Takes up too much memory?
        self.img_clf = CelebaImgClassifier().to(pl_module.device)
        self.text_clf = CelebaTextClassifier().to(pl_module.device)

        self.img_clf.load_state_dict(torch.load(img_weights_path))
        self.text_clf.load_state_dict(torch.load(text_weights_path))

        self.img_clf.eval()
        self.text_clf.eval()

    def _cross_coherence(self, pl_module, batch):
        # Compute Cross Coherence
        model = pl_module.model
        device = pl_module.device

        # Get data
        img, text = [x.to(device) for x in batch["data"]]
        labels = batch["label"].to(device).long()  # [B, n_classes], {0, 1}

        with torch.no_grad():
            # Get cross reconstructions
            # img_recons: [B, 3, 64, 64]
            # text_recons: [B, len_sequence, len(alphabet)], probability vectors
            img_recons, text_recons = model.cross_reconstruct([img, text], mean=True)

            # Get predictions: [B, n_classes], sigmoid output
            img_preds = self.img_clf(img_recons)
            text_preds = self.text_clf(text_recons)

        # Compute AP for each class
        for idx, (img_ap, text_ap) in enumerate(self.aps):
            # Extract pred and target for specified class
            img_pred = img_preds[:, idx]
            text_pred = text_preds[:, idx]

            target = labels[:, idx]

            # Update average precision
            img_ap(img_pred, target)
            text_ap(text_pred, target)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        # Update metric classes
        self._cross_coherence(pl_module, batch)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = {}
        img_ap_sum = 0
        text_ap_sum = 0

        for class_, (img_ap, text_ap) in zip(CELEBA_CLASSES, self.aps):
            img_metric = img_ap.compute()
            text_metric = text_ap.compute()

            # To calculate mean AP
            img_ap_sum += img_metric
            text_ap_sum += text_metric

            metrics[f"val_img_{class_}_coh_AP"] = img_metric
            metrics[f"val_text_{class_}_coh_AP"] = text_metric

        metrics["val_img_coh_mAP"] = img_ap_sum / len(CELEBA_CLASSES)
        metrics["val_text_coh_mAP"] = text_ap_sum / len(CELEBA_CLASSES)

        # FIXME Could be wrong
        pl_module.log_dict(metrics)

        # logger = pl_module.logger.experiment
        # logger.log(metrics, commit=False)
