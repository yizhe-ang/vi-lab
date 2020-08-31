import torch
import torch.nn as nn

from .flows import Planar


class NormalizingFlowModel(nn.Module):

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m, device=x.device)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld

        z = x
        return z, log_det

    def inverse(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det

    # def sample(self, n_samples):
    #     z = self.prior.sample((n_samples,))
    #     x, _ = self.inverse(z)
    #     return x


class PlanarModel(NormalizingFlowModel):
    def __init__(self, dim, n_layers=20):
        flows = [Planar(dim) for _ in range(n_layers)]
        super().__init__(flows)
