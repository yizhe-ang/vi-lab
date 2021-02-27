import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_tsne(z_loc: torch.Tensor, classes: torch.Tensor):
    """
    Parameters
    ----------
    z_loc : torch.Tensor
        [B, Z]
    classes : torch.Tensor
        [B]

    Returns
    -------
    [type]
        Matplotlib fig
    """
    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_loc.detach().cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)
    classes = classes.detach().cpu().numpy()

    fig = plt.figure(figsize=(5, 5))
    for ic in range(10):
        ind_class = classes == ic
        color = plt.cm.Set3(ic)

        plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, color=color)
        # plt.title(title)

    return fig

def plot_pca(z_loc: torch.Tensor, classes: torch.Tensor):
    """
    Parameters
    ----------
    z_loc : torch.Tensor
        [B, Z]
    classes : torch.Tensor
        [B]

    Returns
    -------
    [type]
        Matplotlib fig
    """
    model_pca = PCA(n_components=2, random_state=0)
    z_states = z_loc.detach().cpu().numpy()
    z_embed = model_pca.fit_transform(z_states)
    classes = classes.detach().cpu().numpy()

    fig = plt.figure(figsize=(5, 5))
    for ic in range(10):
        ind_class = classes == ic
        color = plt.cm.Set3(ic)

        plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, color=color)
        # plt.title(title)

    return fig


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
        # HACK include for factorized gaussian
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
    plot_distribution(
        p_dist,
        bounds=((-2, 8), (-5, 17)),  # HACK Hardcoded
        ax=ax,
        num=100,
        n_levels=100,
        exp=True,
        filled=True,
    )
    plot_points(samples, ax=ax)

    return ax
