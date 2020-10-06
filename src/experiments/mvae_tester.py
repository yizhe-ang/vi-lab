from src.models.classifiers import LatentClassifier, SVHN_Classifier, MNIST_Classifier
from pathlib import Path
import torch


class MultimodalVAE_Tester:
    def __init__(self, pl_module):
        # FIXME Ensure desired weights are loaded in pl_module
        self.pl_module = pl_module
        self.pl_module.eval()

        # Init classifiers
        self._init_classifiers()

        # Get dataloaders
        self.test_loader = self.pl_module.datamodule.test_dataloader()

    def _init_classifiers(self):
        # Load pretrained weights
        mnist_weights_path = Path("saved") / "mnist_svhn" / "mnist_model.pt"
        svhn_weights_path = Path("saved") / "mnist_svhn" / "svhn_model.pt"

        self.mnist_net = MNIST_Classifier().to(self.pl_module.device)
        self.svhn_net = SVHN_Classifier().to(self.pl_module.device)

        self.mnist_net.load_state_dict(torch.load(mnist_weights_path))
        self.svhn_net.load_state_dict(torch.load(svhn_weights_path))

        self.mnist_net.eval()
        self.svhn_net.eval()

    def cross_coherence(self):
        print("Evaluating Cross Coherence...")

        model = self.pl_module.model
        device = self.pl_module.device

        dataset_size = len(self.test_loader.dataset)

        corr_m = 0
        corr_s = 0

        with torch.no_grad():
            for batch in self.test_loader:
                mnist, svhn = batch["data"]
                targets = batch["label"]
                mnist, svhn, targets = (
                    mnist.to(device),
                    svhn.to(device),
                    targets.to(device),
                )

                # Get cross reconstructions
                m_recons, s_recons = model.cross_reconstruct(
                    [mnist, svhn], mean=True
                )

                # Get predictions
                m_preds = self.mnist_net(m_recons).argmax(dim=1)
                s_preds = self.svhn_net(s_recons).argmax(dim=1)

                # Evaluate correct reconstructions
                corr_m += (m_preds == targets).sum().item()
                corr_s += (s_preds == targets).sum().item()

        return {
            'cross_coherence_s_m': corr_m / dataset_size,
            'cross_coherence_m_s': corr_s / dataset_size
        }

    def joint_coherence(self, n_samples: int = 10_000):
        print("Evaluating Joint Coherence...")

        model = self.pl_module.model
        corr = 0

        with torch.no_grad():
            # Generate samples
            mnist, svhn = model.sample(n_samples, mean=True)

            # Get predictions
            m_preds = self.mnist_net(mnist).argmax(dim=1)
            s_preds = self.svhn_net(svhn).argmax(dim=1)

            # Evaluate correct samples
            corr += (m_preds == s_preds).sum().item()

        return {
            'joint_coherence': corr / n_samples
        }

    def evaluate(self):
        logger = self.pl_module.logger.experiment

        cross_coherence_results = self.cross_coherence()
        joint_coherence_results = self.joint_coherence()

        # Log results
        logger.log({**cross_coherence_results, **joint_coherence_results})
