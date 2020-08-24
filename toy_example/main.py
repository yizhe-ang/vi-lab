"""To solve the optimization problem KL(Q | P)
using different gradient estimators"""

from collections import deque
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import wandb
import yaml
from torch.distributions.kl import kl_divergence

import losses as losses
import p_dists as p_dists
import q_models as q_models
from viz import plot_samples


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

    # Keep queue of samples for visualization
    samples = deque()
    fig, ax = plt.subplots(figsize=(10, 10))

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

            samples.append(q_model.prev_sample.numpy().copy())
            if len(samples) > 1_000:
                samples.popleft()

            if (i + 1) % 1_000 == 0:
                #ax = plot_samples(np.stack(samples), p_dist, ax)
                #wandb.log({"samples": wandb.Image(ax)})
                print("logged chart")
                print(q_model.loc, q_model.prec)


if __name__ == "__main__":
    # Load config yaml file
    config_path =  Path("config.yaml")
    with config_path.open("r") as f:
        cfg = yaml.load(f)

    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(cfg["seed"])

    main(cfg)
