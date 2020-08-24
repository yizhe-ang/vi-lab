import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

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

class MVNormalDist(nn.Module):
    """Parameterized Normal Distribution"""

    def __init__(self, n_dim):
        super().__init__()

        self.loc = nn.Parameter(torch.randn(n_dim))
        self.prec_factor = nn.Parameter(torch.eye(n_dim))
        # Cache latest sample
        self.prev_sample = torch.randn(n_dim)

    def get_dist(self):
        self.prec = self.prec_factor @ self.prec_factor.t() + 1e-6 * torch.eye(2)
        return MultivariateNormal(self.loc, precision_matrix=self.prec)
