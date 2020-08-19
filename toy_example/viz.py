import numpy as np
import torch


def plot_distribution(
    distribution, bounds, ax, num=50, n_levels=None, filled=False, exp=False
):
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    x, y = np.meshgrid(
        np.linspace(x_min, x_max, num=num), np.linspace(y_min, y_max, num=num)
    )
    s = x.shape
    xy = np.stack([x.reshape(-1), y.reshape(-1)], axis=1)
    z = (
        distribution.log_prob(torch.tensor(xy, dtype=torch.float32))
        .sum(-1)
        .numpy()
        .reshape(s)
    )
    if exp:
        z = np.exp(z)

    plot = ax.contourf if filled else ax.contour
    r = plot(x, y, z, n_levels, cmap="binary")
    return ax, r


def plot_points(xs, ax, i=0, j=1, color=True):
    n_samples, _ = xs.shape
    c = np.arange(n_samples) if color else None
    ax.scatter(xs[:, i], xs[:, j], s=5, c=c)
    return ax


def plot_samples(samples, p_dist, ax):
    # FIXME Determine bounds; hardcoding distribution values
    min_1 = min(samples[:, 0].min(), 3)
    max_1 = max(samples[:, 0].max(), 3)
    min_2 = min(samples[:, 1].min(), 6)
    max_2 = max(samples[:, 1].max(), 6)

    plot_distribution(
        p_dist,
        bounds=((0, 6), (-8, 20)),
        ax=ax,
        num=100,
        n_levels=100,
        exp=True,
        filled=True,
    )
    plot_points(samples, ax=ax)

    return ax
