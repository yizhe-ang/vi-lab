import itertools
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import (
    DmolNet,
    draw_gaussian_diag_samples,
    gaussian_analytical_kl,
    get_1x1,
    get_3x3,
)


class Block(nn.Module):
    def __init__(
        self,
        in_width: int,
        middle_width: int,
        out_width: int,
        down_rate: int = None,
        residual=False,
        use_3x3=True,
        zero_last=False,
    ):
        """Bottleneck Residual Block

        Parameters
        ----------
        in_width : int
            Number of input features
        middle_width : int
            Number of intermediate features
        out_width : int
            Number of output features
        down_rate : int, optional
            Input downsampling rate, by default None
        residual : bool, optional
            Whether to include residual connection, by default False
        use_3x3 : bool, optional
            Whether to use 3x3 conv or 1x1 conv as intermediate layers, by default True
            Set to False, when processing 1x1 or 2x2 input patches
        zero_last : bool, optional
            Whether to initialize weights to be zero in the last 1x1 conv layer, by default False
            Set to True when outputting parameters for normal prior?
        """
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual

        # Conv layers
        # Downsample number of features (bottleneck)
        self.c1 = get_1x1(in_width, middle_width)
        # Two intermediate conv layers
        self.c2 = (
            get_3x3(middle_width, middle_width)
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        self.c3 = (
            get_3x3(middle_width, middle_width)
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        # Upsample number of features
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        """
        1. Process through residual block
        2. Avg pooling downsampling (if specified)

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        # Pre-activation
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))

        # Residual connection
        out = x + xhat if self.residual else xhat

        # Downsampling
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out


def parse_layer_string(s):
    # `enc_config`: "32x5,32d2,16x3,o,16d2,8x3,8d2,4x3,o"
    # `o` means output activation of previous layer

    # `dec_config`: "o,4x2,8m4,8x3,16m8,o,16x5,32m16,32x10"
    # `o` means subsequent layer processes latent vectors

    # List of layers specified by
    # (input_res, downsample rate)
    # (input_res, reso of previous layer)
    layers = []
    for ss in s.split(","):
        # Number of conv layers (same res)
        if "x" in ss:
            res, num = ss.split("x")
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]

        elif ss == "o":
            layers.append(True)

        # Upsampling layer
        elif "m" in ss:
            res, mixin = [int(a) for a in ss.split("m")]
            layers.append((res, mixin))

        # Downsampling layer
        elif "d" in ss:
            res, down_rate = [int(a) for a in ss.split("d")]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))

    return layers


class Encoder(nn.Module):
    """Layers of Bottleneck Residual Blocks"""

    def __init__(
        self,
        width: int = 256,
        image_channels: int = 3,
        enc_config: str = "32x5,32d2,16x3,o,16d2,8x3,8d2,4x3,o",
        bottleneck_multiple: float = 0.25,
        scale_init_weights=False,
    ):
        super().__init__()

        # First conv for input image
        self.in_conv = get_3x3(image_channels, width)
        self.width = width

        # Indices of layer activations to output
        self.output_layers = []
        enc_blocks = []
        # Layer configuration
        blocks_config = parse_layer_string(enc_config)
        # Bottleneck residual block for each layer
        for block in blocks_config:
            # If output indicator
            if block is True:
                # To output activation of previous layer
                self.output_layers.append(len(enc_blocks) - 1)

            else:
                res, down_rate = block
                use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
                enc_blocks.append(
                    Block(
                        width,
                        int(width * bottleneck_multiple),
                        width,
                        # Whether to downsample
                        down_rate=down_rate,
                        residual=True,
                        use_3x3=use_3x3,
                    )
                )

        # Get depth of network
        n_blocks = len([b for b in blocks_config if b is not True])

        if scale_init_weights:
            for b in enc_blocks:
                # Scale the initial weights of the final conv layer in each
                # residual bottleneck block
                # (inversely proportional to the depth of the network)
                b.c4.weight.data *= np.sqrt(1 / n_blocks)

        self.enc_blocks = nn.ModuleList(enc_blocks)

    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        Dict[int, torch.Tensor]
            Dictionary of output activations, queried by resolution
        """
        # FIXME Why the need to permute channels?
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.in_conv(x)

        # Output activations, queried by resolution
        activations = {}

        for i, block in enumerate(self.enc_blocks):
            x = block(x)

            # If outputting activation at this layer
            if i in self.output_layers:
                res = x.shape[2]
                activations[res] = x

            # x = (
            #     x
            #     if x.shape[1] == self.widths[res]
            #     # FIXME Why the need to pad number of channels?
            #     else pad_channels(x, self.widths[res])
            # )

        return activations


class LatentDecBlock(nn.Module):
    def __init__(
        self,
        res: int,
        mixin: Optional[int],
        n_blocks: int,
        width=256,
        zdim=16,
        bottleneck_multiple=0.25,
        scale_init_weights=False,
    ):
        super().__init__()

        # Input resolution
        self.base = res
        # Resolution of previous layer
        self.mixin = mixin
        # Number of features for activations
        self.width = width
        # Number of features for latent
        self.zdim = zdim

        use_3x3 = res > 2
        # Bottleneck number of features
        cond_width = int(width * bottleneck_multiple)

        # Encode concat([encoder act, decoder act]) into parameters for q(z|x)
        self.enc = Block(
            width * 2, cond_width, zdim * 2, residual=False, use_3x3=use_3x3
        )

        # Encode into parameters for p(z),
        # as well as activations for downstream processing
        self.prior = Block(
            width,
            cond_width,
            # Parameters for p(z) + downstream activations
            zdim * 2 + width,
            residual=False,
            use_3x3=use_3x3,
            # Why initialize weights to 0?
            # Output parameters closer to standard normal prior?
            zero_last=True,
        )

        # Encode z vector to merge with rest of the network
        self.z_proj = get_1x1(zdim, width)
        if scale_init_weights:
            self.z_proj.weight.data *= np.sqrt(1 / n_blocks)

        # Final layer, to process previous z vector with the network
        self.resnet = Block(width, cond_width, width, residual=True, use_3x3=use_3x3)
        if scale_init_weights:
            self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)

        # FIXME Why need to wrap into a function?
        self.z_fn = lambda x: self.z_proj(x)

    def sample(self, x, acts):
        # Encode concat([encoder act, decoder act])
        # into parameters for q(z_i | z<i, x)
        qm, qv = self.enc(torch.cat([x, acts], dim=1)).chunk(2, dim=1)

        # Encode into parameters for p(z),
        # as well as activations for downstream processing
        feats = self.prior(x)
        pm, pv, xpp = (
            feats[:, : self.zdim, ...],
            feats[:, self.zdim : self.zdim * 2, ...],
            feats[:, self.zdim * 2 :, ...],
        )
        x = x + xpp

        # Sample from q
        z = draw_gaussian_diag_samples(qm, qv)
        # Compute KL(q | p)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)

        return z, x, kl

    def sample_uncond(self, x, t=None, lvs=None):
        n, c, h, w = x.shape
        # Encode into parameters for p(z_i | z<i),
        # as well as activations for downstream processing
        feats = self.prior(x)
        pm, pv, xpp = (
            feats[:, : self.zdim, ...],
            feats[:, self.zdim : self.zdim * 2, ...],
            feats[:, self.zdim * 2 :, ...],
        )
        x = x + xpp

        # If z vectors are specific instead (don't sample)
        if lvs is not None:
            z = lvs
        else:
            # `t`: sampling variance temperature
            if t is not None:
                pv = pv + torch.ones_like(pv) * np.log(t)

            # Sample from prior
            z = draw_gaussian_diag_samples(pm, pv)

        return z, x

    def get_inputs(self, xs, activations):
        # Get corresponding encoder activation
        acts = activations[self.base]
        try:
            # Get previous decoder activation
            x = xs[self.base]
        # If no initial bias parameters, set to activation to zero
        except KeyError:
            x = torch.zeros_like(acts)
        # Repeat batch dimension to match encoder activations
        if acts.shape[0] != x.shape[0]:
            x = x.repeat(acts.shape[0], 1, 1, 1)
        return x, acts

    def forward(self, xs, activations, get_latents=False):
        """Forward pass, combining with encoder activations to compute
        parameters for q(z_i | z<i, x)

        1. Nearest-neighbour upsampling (if specified)
        2. In "parallel",
            a. Encode [encoder act, decoder act] into q(z_i | z<i, x)
            b. Encode [decoder act] into parameters for p(z_i | z<i),
               as well as downstream activations
        3. Sample z from q
        4. Compute KL(q | p)
        5. Encode and process sampled z with the rest of the network

        Parameters
        ----------
        xs : Dict[res, torch.Tensor]
            All input activations from decoder network
        activations :
            All activations from encoder network
        get_latents : bool, optional
            Return latents, by default False

        Returns
        -------
        [type]
            Decoder activations, {z latents, kl term}
        """
        # Get relevant encoder activation, and decoder input activation
        x, acts = self.get_inputs(xs, activations)

        # If upsampling
        if self.mixin is not None:
            # Nearest neighbour upsampling
            # `x` is a tensor of zeros here
            x = x + F.interpolate(
                xs[self.mixin][:, : x.shape[1], ...],
                scale_factor=self.base // self.mixin,
            )

        # Sample from q, compute KL(q | p)
        z, x, kl = self.sample(x, acts)

        # Encode z vector to merge with rest of the network
        x = x + self.z_fn(z)
        # Final layer, to process previous z vector with the network
        x = self.resnet(x)

        # Update decoder activations
        xs[self.base] = x

        if get_latents:
            return xs, dict(z=z.detach(), kl=kl)
        return xs, dict(kl=kl)

    def forward_uncond(self, xs, t: float = None, lvs=None):
        """Forward pass (unconditioned on encoder activations),
        sampling from prior p(z_i | z<i) directly

        1. Nearest-neighbour upsampling (if specified)
        2. Encode [decoder act] into parameters for p(z_i | z<i),
           as well as downstream activations
        4. If z latents not specified,
           sample z from p with specified temperature
        5. Encode and process sampled z with the rest of the network

        Parameters
        ----------
        xs : Dict[int, torch.Tensor]
            All input activations from decoder network, queried by resolution
        t : float
            Sampling variance temperature
        lvs : torch.Tensor
            Specify z latents to decode from

        Returns
        -------
        [type]
            Decoder activations
        """
        # Get latest input activation (of appropriate resolution)
        try:
            x = xs[self.base]
        except KeyError:
            # Initialize to zero if not found
            ref = xs[list(xs.keys())[0]]
            x = torch.zeros(
                dtype=ref.dtype,
                size=(ref.shape[0], self.width, self.base, self.base),
                device=ref.device,
            )
        # If upsampling
        if self.mixin is not None:
            # Nearest neighbour upsampling
            # `x` is either bias term, or tensor or zeros
            x = x + F.interpolate(
                xs[self.mixin][:, : x.shape[1], ...],
                scale_factor=self.base // self.mixin,
            )

        # Get z latent, and downstream activations
        z, x = self.sample_uncond(x, t, lvs=lvs)

        # Encode z vector to merge with rest of the network
        x = x + self.z_fn(z)
        # Final layer, to process previous z vector with the network
        x = self.resnet(x)

        # Update decoder activations
        xs[self.base] = x

        return xs


class DecBlock(Block):
    def __init__(
        self,
        res: int,
        mixin: Optional[int],
        n_blocks: int,
        width=256,
        bottleneck_multiple=0.25,
        scale_init_weights=False,
    ):
        super().__init__(
            width,
            int(width * bottleneck_multiple),
            width,
            residual=True,
            use_3x3=res > 2,
        )
        # Input resolution
        self.base = res
        # Resolution of previous layer
        self.mixin = mixin

        if scale_init_weights:
            self.c4.weight.data *= np.sqrt(1 / n_blocks)

    def forward(self, xs):
        """
        1. Nearest-neighbour upsampling (if specified)
        2. Process through residual block

        Parameters
        ----------
        xs : Dict[int, torch.Tensor]
            All input activations from decoder network, queried by resolution

        Returns
        -------
        Dict[int, torch.Tensor]
            All input activations from decoder network, queried by resolution
        """
        # Get latest input activation (of appropriate resolution)
        try:
            x = xs[self.base]
        except KeyError:
            # Initialize to tensor of zeros if not found
            ref = xs[list(xs.keys())[0]]
            x = torch.zeros(
                dtype=ref.dtype,
                size=(ref.shape[0], self.width, self.base, self.base),
                device=ref.device,
            )
        # If upsampling
        if self.mixin is not None:
            # Nearest neighbour upsampling
            # `x` is either bias term, or tensor or zeros
            x = x + F.interpolate(
                xs[self.mixin][:, : x.shape[1], ...],
                scale_factor=self.base // self.mixin,
            )

        # Process through resnet
        x = super().forward(x)

        # Update decoder activations
        xs[self.base] = x

        return xs


class Decoder(nn.Module):
    def __init__(
        self,
        dec_config: str = "o,4x2,8m4,8x3,o,16m8,16x5,32m16,32x10",
        width=256,
        zdim=16,
        bottleneck_multiple=0.25,
        scale_init_weights=False,
        no_bias_above=64,
        num_mixtures=10,
        image_size=32,
    ):
        super().__init__()
        # Set of layer resolutions
        resos = set()

        self.width = width
        self.image_size = image_size

        # Layer configuration
        blocks_config = parse_layer_string(dec_config)
        n_blocks = len([b for b in blocks_config if b is not True])

        block_kwargs = {
            "n_blocks": n_blocks,
            "width": width,
            "bottleneck_multiple": bottleneck_multiple,
            "scale_init_weights": scale_init_weights,
        }

        # Initialize decoder blocks
        dec_blocks = []
        next_latent = False  # Flag to indicate whether next block is a latent block

        for block in blocks_config:
            # Next block will be a latent block
            if block is True:
                next_latent = True
                continue

            res, mixin = block
            if next_latent:
                dec_blocks.append(LatentDecBlock(res, mixin, zdim=zdim, **block_kwargs))
                next_latent = False

            else:
                dec_blocks.append(DecBlock(res, mixin, **block_kwargs))

            # Store all available layer resolutions
            resos.add(res)

        self.resolutions = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)

        # Bias parameters / initial activations for each resolution
        self.bias_xs = nn.ParameterList(
            [
                # Initialized as zeros
                nn.Parameter(torch.zeros(1, width, res, res))
                for res in self.resolutions
                if res <= no_bias_above
            ]
        )

        # Output data modeled by a discretized mixture of logistics
        self.out_net = DmolNet(width, num_mixtures)

        # Scale and bias parameters for final output data
        self.gain = nn.Parameter(torch.ones(1, width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, width, 1, 1))
        self.final_fn = lambda x: x * self.gain + self.bias

    def forward(self, activations, get_latents=False):
        """
        Parameters
        ----------
        activations : Dict[int, torch.Tensor]
            Encoder activations queried by resolution
        get_latents : bool, optional
            Whether to return z latents, by default False

        Returns
        -------
        [type]
            Final layer activation, {z latents, kl term}
            Each kl term:  [B, zdim, H, W]
            Each z latent: [B, zdim, H, W]
        """
        stats = []
        # Get bias parameters / initial activations for each resolution
        xs = {a.shape[2]: a for a in self.bias_xs}

        for block in self.dec_blocks:
            # Different inputs for different blocks
            if type(block) is LatentDecBlock:
                xs, block_stats = block(xs, activations, get_latents=get_latents)
                stats.append(block_stats)
            else:
                xs = block(xs)

        # Run final output through final scale and bias parameters
        xs[self.image_size] = self.final_fn(xs[self.image_size])

        return xs[self.image_size], stats

    def forward_uncond(self, n: int, t: Union[float, List[float]] = None):
        """
        Parameters
        ----------
        n : int
            Number of samples
        t : Union[float, List[float]]
            Sampling variance temperature/s

        Returns
        -------
        torch.Tensor
            Final layer activation
        """
        # Get bias parameters / initial activations for each resolution
        xs = {}
        for bias in self.bias_xs:
            # Repeat by number of samples `n`
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)

        latent_block_idx = 0
        for block in self.dec_blocks:
            if type(block) is LatentDecBlock:
                # Check whether same temperature for all latent layers
                if type(t) is list:
                    temp = t[latent_block_idx]
                    latent_block_idx += 1
                else:
                    temp = t
                xs = block.forward_uncond(xs, temp)
            else:
                xs = block(xs)

        # Run final output through final scale and bias parameters
        xs[self.image_size] = self.final_fn(xs[self.image_size])

        return xs[self.image_size]

    def forward_manual_latents(
        self, n: int, latents: List[torch.Tensor], t: float = None
    ) -> None:
        """Manually specify z latents

        Parameters
        ----------
        n : int
            Number of samples
        latents : List[torch.Tensor]
            [description]
        t : float, optional
            [description], by default None

        Returns
        -------
        torch.Tensor
            Final layer activation
        """
        # Get bias parameters / initial activations for each resolution
        xs = {}
        for bias in self.bias_xs:
            # Repeat by number of samples `n`
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)

        latent_block_idx = 0
        for block in self.dec_blocks:
            if type(block) is LatentDecBlock:
                if latent_block_idx < len(latents):
                    lvs = latents[latent_block_idx]
                    latent_block_idx += 1
                else:
                    lvs = None

                xs = block.forward_uncond(xs, t, lvs=lvs)
            else:
                xs = block(xs)

        # Run final output through final scale and bias parameters
        xs[self.image_size] = self.final_fn(xs[self.image_size])

        return xs[self.image_size]
