import torch
import torch.nn as nn


class FeedForwardEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 2048,
        dropout_ratio: float = 0.1,
    ):
        super().__init__()

        self._encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._encoder(x)
