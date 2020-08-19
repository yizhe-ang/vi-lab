import torch
import torch.nn as nn


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

    def reconstruct(self):
        pass


class Distribution:
    """Abstract class for unnormalized distribution"""

    def log_prob(self, x: torch.tensor) -> torch.tensor:
        """Computes vectorized log of unnormalized log density

        Parameters
        ----------
        x : torch.tensor
            [B, D], B points at which we compute log density

        Returns
        -------
        torch.tensor
            [B], \log \hat{\pi}(x)
        """
        raise NotImplementedError

    def grad_log_density(self, x: torch.tensor) -> torch.tensor:
        """Computes vectorized gradient \nabla_x \log \pi(x)

        Parameters
        ----------
        x : torch.tensor
            [B, D], Point at which we compute \nabla \log \pi

        Returns
        -------
        torch.tensor
            [B, D], Gradients of log density
        """
        # FIXME why clone and not detach?
        x = x.clone().requires_grad_()
        logp = self.log_density(x)
        logp.sum().backward()

        return x.grad


class Proposal:
    """Abstract class for proposal"""

    def sample(self, x: torch.tensor) -> torch.tensor:
        """Computes vectorized sample from proposal q(x'|x)

        Parameters
        ----------
        x : torch.tensor
            [B, D], Current point from which we propose

        Returns
        -------
        torch.tensor
            [B, D], New points
        """

        raise NotImplementedError

    def log_density(self, x: torch.tensor, x_prime: torch.tensor) -> torch.tensor:
        """Computes vectorized log of unnormalized log density

        Parameters
        ----------
        x : torch.tensor
            [B, D], B points at which we compute log density
        x_prime : torch.tensor
            [B, D]

        Returns
        -------
        torch.tensor
            [B], \log q(x'|x)
        """
        raise NotImplementedError

