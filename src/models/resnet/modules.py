from typing import Dict, List, Optional, Union

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .helpers import (
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

    def forward(self, x):
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
        # If upsampling
        if self.mixin is not None:
            # Nearest neighbour upsampling
            # `x` is either bias term, or tensor or zeros
            x = F.interpolate(
                x,
                scale_factor=self.base // self.mixin,
            )

        # Process through resnet
        x = super().forward(x)

        return x


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


class ResNetEncoder(nn.Module):
    """Layers of Bottleneck Residual Blocks"""

    def __init__(
        self,
        latent_dim,
        width: int = 32,
        image_channels: int = 3,
        enc_config: str = "32x1,32d2,16x1,16d2,8x1,8d2,4x1",
        bottleneck_multiple: float = 0.25,
        scale_init_weights=False,
    ):
        super().__init__()

        # First conv for input image
        # self.in_conv = get_3x3(image_channels, width)
        self.in_conv = nn.Conv2d(image_channels, width, kernel_size=1)

        enc_blocks = []
        # Layer configuration
        blocks_config = parse_layer_string(enc_config)
        # Bottleneck residual block for each layer
        for block in blocks_config:
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
        n_blocks = len(blocks_config)

        if scale_init_weights:
            for b in enc_blocks:
                # Scale the initial weights of the final conv layer in each
                # residual bottleneck block
                # (inversely proportional to the depth of the network)
                b.c4.weight.data *= np.sqrt(1 / n_blocks)

        self.enc_blocks = nn.ModuleList(enc_blocks)

        final_res = blocks_config[-1][0]
        self.final_dim = final_res * final_res * width
        # Final linear layer
        self.out_linear = nn.Linear(self.final_dim, latent_dim * 2)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            [B, C, H, W]

        Returns
        -------
        torch.Tensor
            [B, output_dim]
        """
        x = self.in_conv(x)

        for block in self.enc_blocks:
            x = block(x)

        x = x.reshape(-1, self.final_dim)
        x = F.gelu(x)
        x = self.out_linear(x)

        return x


class ResNetDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        width=256,
        image_channels=3,
        dec_config="4x1,8m4,8x1,16m8,16x1,32m16,32x1",
        bottleneck_multiple=0.25,
        scale_init_weights=False,
    ):
        super().__init__()

        self.width = width

        # Layer configuration
        blocks_config = parse_layer_string(dec_config)
        n_blocks = len(blocks_config)

        block_kwargs = {
            "n_blocks": n_blocks,
            "width": width,
            "bottleneck_multiple": bottleneck_multiple,
            "scale_init_weights": scale_init_weights,
        }

        # Initialize decoder blocks
        dec_blocks = []

        for block in blocks_config:
            res, mixin = block
            dec_blocks.append(DecBlock(res, mixin, **block_kwargs))

        self.dec_blocks = nn.ModuleList(dec_blocks)

        self.initial_res = blocks_config[0][0]
        self.in_linear = nn.Linear(
            latent_dim, self.initial_res * self.initial_res * width
        )
        self.out_conv = nn.Conv2d(width, image_channels, kernel_size=1)

    def forward(self, x):
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
        x = self.in_linear(x).reshape(
            -1, self.width, self.initial_res, self.initial_res
        )

        for block in self.dec_blocks:
            x = block(x)

        x = F.gelu(x)
        x = self.out_conv(x)

        return x
