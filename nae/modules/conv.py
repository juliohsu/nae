"""Convolutional layers wrappers and utilities."""

import math
import typing as tp
import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, spectral_norm

from .conv_norm import ConvLayerNorm


CONV_NORMALIZATIONS = frozenset(
    [
        "none",
        "weight_norm",
        "spectral_norm",
        "time_layer_norm",
        "layer_norm",
        "time_group_norm",
    ]
)


def apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    elif norm == "spectral_norm":
        return spectral_norm(module)
    else:
        # We have already check if "norm" is in CONV_NORMALIZATIONS,
        # so any other choices doesn't need reparametrization.
        return module


def get_norm_module(
    module: nn.Module, causal: bool, norm: str = "none", **norm_kwargs
) -> nn.Module:
    """Return the proper normalization module. If causal is true, this will ensure returned model is causal,
    or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == "layer_norm":
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == "time_group_norm":
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """See `pad_for_conv1d`"""
    length = x.shape(-1)
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames - 1) * stride) + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
):
    """Add extra padding for the 1d convolution"""
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))


def pad1d(
    x: torch.Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = "zero",
    value: float = 0.0,
):
    """Tiny wrapper around F.pad, just to allow for reflect padding around small input.
    If this is the case, we insert extra 0 padding to the right before reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        pass
    else:
        return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization around this convolution
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        return self.norm(x)


class NormConv2d(nn.Module):
    """Wrapper around Conv2d and normalization around this convolution,
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal=False, norm=norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        return self.norm(x)


class NormConvTranspose1d(nn.Module):
    """Wrapper around ConvTranspose1d and normalization around this convolution,
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs))
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        return self.norm(x)


class NormConvTranspose2d(nn.Module):
    """Wrapper ConvTranspose2d and normalization around this convolution,
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.ConvTranspose2d(*args, **kwargs))
        self.norm = get_norm_module(self.conv, causal=False, norm=norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        return self.norm(x)


class SConv1d(nn.Module):
    """Conv1 with some built-in handling of asymmetric or causal padding and normalization."""

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        group: int = 1,
        bias: int = 1,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        pad_mode: str = "reflect",
    ):
        super().__init__()
        # warn user between stride and dilation setup
        if stride > 1 and dilation > 1:
            warnings.warn(
                "SConv1d has been initialized with stride > 1 and dilation > 1"
                f"(kernel_size: {kernel_size}, stride: {stride}, dilation: {dilation})."
            )
        self.causal = causal
        self.pad_mode = pad_mode
        self.conv = NormConv1d(
            in_chs,
            out_chs,
            kernel_size,
            stride,
            dilation,
            group,
            bias,
            causal,
            norm,
            norm_kwargs,
        )

    def forward(self, x):
        B, C, T = x.shape
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (
            kernel_size - 1
        ) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(
            x, kernel_size, stride, padding_total
        )
        if self.causal:
            # left padding for causal
            x = pad1d(x, (padding_total, extra_padding), self.pad_mode)
        else:
            # asymmetric paddings required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), self.pad_mode)
        return self.conv(x)


class SConvTranspose1d(nn.Module):
    "ConvTranspose1d with some built-in asymmetric or causal padding and normalization."

    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size,
        stride: int = 1,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        trim_right_ratio: float = 1.0,
    ):
        super().__init__()
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        self.convtr = NormConvTranspose1d(
            in_chs, out_chs, kernel_size, stride, causal, norm, norm_kwargs
        )
        assert (
            self.causal or self.trim_right_ratio == 1.0
        ), "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0.0 and self.trim_right_ratio <= 1.0

    def forward(self, x):
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride

        x = self.convtr(x)

        # We will only trim the fixed padding. Extra padding from `pad_for_conv1d`
        # will only to be removed at the very end, when keeping only the right length for the output.
        # As removing it here would required to pass the length at the matching layer in the encoder.
        if self.causal:
            # Trim the padding on the right according the specified ratio,
            # if trim_right_ratio is 1.0, trim everything from the right.
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            x = unpad1d(x, (padding_left, padding_right))
        else:
            # asymmetric paddings required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = unpad1d(x, (padding_left, padding_right))

        return x
