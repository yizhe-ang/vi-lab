import torch.nn as nn


def get_conv(
    in_dim,
    out_dim,
    kernel_size,
    stride,
    padding,
    zero_bias=True,
    zero_weights=False,
    groups=1,
    scaled=False,
):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_3x3(
    in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False
):
    return get_conv(
        in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups, scaled=scaled
    )


def get_1x1(
    in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False
):
    return get_conv(
        in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups, scaled=scaled
    )
