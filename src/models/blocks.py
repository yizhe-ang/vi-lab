from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        if lambd is None:
            lambd = lambda x: x
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Flatten(nn.Module):
    def forward(self, x):
        # Preserve batch size
        return x.view(x.shape[0], -1)


class MLP(nn.Sequential):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_units: List[int],
        activation: str = "ReLU",
        in_lambda=None,
        out_lambda=None,
    ):
        layers = []

        if in_lambda:
            layers.append(LambdaLayer(in_lambda))

        for in_size, out_size in zip([input_size] + hidden_units[:-1], hidden_units):
            layers.append(nn.Linear(in_size, out_size))
            # Get activation layer from nn module
            layers.append(getattr(nn, activation)())
        layers.append(nn.Linear(hidden_units[-1], output_size))

        if out_lambda:
            layers.append(LambdaLayer(out_lambda))

        super().__init__(*layers)
