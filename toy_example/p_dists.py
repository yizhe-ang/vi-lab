import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal


normal_dist = Normal(loc=torch.tensor([3.0, 6.0]), scale=torch.tensor([1.0, 5.0]))

mvnormal_dist = MultivariateNormal(
    loc=torch.tensor([1, 1]), precision_matrix=torch.tensor([[1.1, 0.9], [0.9, 1.1]])
)
