"""To solve the optimization problem KL(Q | P)
using different gradient estimators"""

from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import wandb
import yaml
from torch.distributions.kl import kl_divergence

import toy_example.losses as losses
import toy_example.p_dists as p_dists
import toy_example.q_models as q_models


def main(cfg):
    # Initialize logging
    wandb.init(name=cfg["name"], config=cfg, project="vi-toy")

    p_dist = getattr(p_dists, cfg["p_dist"])
    q_model = getattr(q_models, cfg["q_model"])(n_dim=2)
    # Log gradients
    wandb.watch(q_model)

    optimizer = getattr(optim, cfg["optimizer"])(
        q_model.parameters(), **cfg["optimizer_args"]
    )

    loss_func = getattr(losses, cfg["loss"])
    if cfg["loss_args"]:
        loss_func = partial(loss_func, **cfg["loss_args"])

    # device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    for i in range(cfg["n_iter"]):
        optimizer.zero_grad()

        # loss = reparam_kl(q_model, p_dist)
        loss = loss_func(q_model, p_dist)

        loss.backward()
        optimizer.step()

        # Log metrics
        with torch.no_grad():
            kl = kl_divergence(q_model.get_dist(), p_dist).mean().item()

            wandb.log({"kl_divergence": kl})


if __name__ == "__main__":
    # Load config yaml file
    config_path = Path("toy_example") / "config.yaml"
    with config_path.open("r") as f:
        cfg = yaml.load(f)

    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(cfg["seed"])

    main(cfg)
