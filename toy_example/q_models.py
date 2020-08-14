import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class NormalDist(nn.Module):
    """Parameterized Normal Distribution"""

    def __init__(self, n_dim):
        super().__init__()

        self.loc = nn.Parameter(torch.randn(n_dim))
        self.log_scale = nn.Parameter(torch.zeros(n_dim))
        # Cache latest sample
        self.prev_sample = torch.randn(n_dim)

    def get_dist(self):
        return Normal(self.loc, self.log_scale.exp())
