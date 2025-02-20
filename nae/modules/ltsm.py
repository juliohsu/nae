"""LTSM layer model."""

import torch.nn as nn


class LTSM(nn.Module):
    """LTSM without worrying about hidden state, nor the input data, but expects input as convolutional layout."""

    def __init__(self, dimension: int, n_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.ltsm = nn.LSTM(dimension, dimension, num_layers=n_layers)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.ltsm(x)
        if self.skip:
            y += x
        return y.permute(1, 2, 0)
