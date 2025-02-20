"""Normalization modules."""

import typing as tp
from einops import rearrange as r

import torch
import torch.nn as nn


class ConvLayerNorm(nn.LayerNorm):
    """Convolutional-friendly LayerNorm that moves channel to last dimension before running normalization,
    and moves them back to original position right after.
    """

    def __init__(
        self, normalized_shape: tp.Union[int, tp.List[int], torch.Size], **kwargs
    ):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = r(x, "b ... t -> b t ...")
        super().forward(x)
        x = r(x, "b t ... -> b ... t")
        return
