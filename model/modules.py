import torch
import torch.nn as nn


class FeedForwardEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 512,
        dropout_ratio: float = 0.1,
    ):
        super().__init__()

        self._bn = nn.BatchNorm1d(hidden_channels)
        self._in_proj = nn.Linear(in_channels, hidden_channels, bias=False)
        self._out_proj = nn.Linear(hidden_channels, out_channels, bias=False)
        self._dropout = nn.Dropout(dropout_ratio)
        self._activation = nn.LeakyReLU(0.2)

        self._out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, x_shape = x.view(x.shape[0], -1, x.shape[-1]), x.shape
        x = self._in_proj(x)
        x = self._bn(x.transpose(1, 2)).transpose(1, 2)
        x = self._activation(x)
        x = self._dropout(x)
        x = self._out_proj(x)
        shape = list(x_shape[:-1]) + [self._out_channels]
        return x.view(*shape)
