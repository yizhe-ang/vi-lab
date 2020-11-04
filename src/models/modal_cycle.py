import torch.nn as nn
from src.models.dists import standard_flow
from typing import List
from nflows.distributions import Distribution


class ModalCycle(nn.Module):
    def __init__(
        self,
        priors: List[Distribution],
        approximate_posteriors: List[Distribution],
        likelihoods: List[Distribution],
        inputs_encoders: List[nn.Module]
    ):
        super().__init__()
        self.priors = nn.ModuleList(priors)
        self.approximate_posteriors = nn.ModuleList(approximate_posteriors)
        self.likelihoods = nn.ModuleList(likelihoods)
        self.inputs_encoders = nn.ModuleList(inputs_encoders)


