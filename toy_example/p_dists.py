import torch
from torch.distributions.normal import Normal


normal_dist = Normal(
    loc=torch.tensor([1.0, 2.0]),
    scale=torch.tensor([1.0, 5.0])
)