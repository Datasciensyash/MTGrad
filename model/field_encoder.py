import math
import random

import torch
import torch.nn as nn

from model.modules import FeedForwardEncoder


class AbsolutePositionalEncoding(nn.Module):
    def __init__(
        self, in_channels: int, max_length: int = 128, requires_grad: bool = True
    ):
        super(AbsolutePositionalEncoding, self).__init__()
        self._in_channels = in_channels
        self._max_length = max_length

        if requires_grad:
            positional_encoding = self.initialize_random_encoding_table()
        else:
            positional_encoding = self.initialize_sinusoid_encoding_table()

        self._positional_encoding = nn.Parameter(
            positional_encoding, requires_grad=requires_grad
        )
        self._alpha = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

        self.register_parameter("_positional_encoding", self._positional_encoding)
        self.register_parameter("_alpha", self._alpha)

    def initialize_sinusoid_encoding_table(self) -> torch.Tensor:
        position = torch.arange(self._max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self._in_channels, 2) * (-math.log(10000.0) / self._in_channels))
        encoding = torch.zeros(self._max_length, self._in_channels)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding

    def initialize_random_encoding_table(self) -> torch.Tensor:
        positional_encoding = torch.empty((self._max_length, self._in_channels))
        nn.init.kaiming_uniform_(positional_encoding, mode="fan_in")
        return positional_encoding

    def forward(
        self, input_tensor: torch.Tensor, random_slice: bool = False
    ) -> torch.Tensor:
        tensor_length = input_tensor.shape[2]  # (B, H, W, C) -> W
        start_index = (
            0
            if not random_slice
            else random.randint(0, self._max_length - tensor_length)
        )
        encoding = self._positional_encoding[start_index : start_index + tensor_length]
        # (B, H, W, C) + (W, C) -> (B, H, W, C)
        return self._alpha * encoding + input_tensor


class PeriodEncoder(nn.Module):
    def __init__(self, out_channels: int, log: bool = False):
        super(PeriodEncoder, self).__init__()
        self._log = log
        self._encoder = nn.Sequential(nn.Linear(1, out_channels), nn.LeakyReLU(0.2))
        self._alpha = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

        self.register_parameter("_alpha", self._alpha)

    def forward(self, periods: torch.Tensor) -> torch.Tensor:
        periods = periods.unsqueeze(-1)
        periods = periods if not self._log else torch.log(periods)
        return self._encoder(periods) * self._alpha


class FieldEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        pos_enc_max_length: int = 1024,
        pos_enc_requires_grad: bool = True,
        period_enc_log: bool = False,
    ):
        super(FieldEncoder, self).__init__()

        self._positional_encoding = AbsolutePositionalEncoding(
            hidden_channels,
            max_length=pos_enc_max_length,
            requires_grad=pos_enc_requires_grad,
        )
        self._period_encoding_log = period_enc_log

        self._field_projection = FeedForwardEncoder(2, hidden_channels)

    def forward(self, field: torch.Tensor, periods: torch.Tensor) -> torch.Tensor:

        # Add periods data to field
        field = field.squeeze(1) if field.ndim == 4 else field
        periods = periods.unsqueeze(2).repeat(1, 1, field.shape[-1])
        periods = torch.log(periods) if self._period_encoding_log else periods
        field = torch.stack([field, periods], dim=3)

        # Encode field
        field = self._field_projection(field)

        # Add positional encoding
        field = self._positional_encoding(field)

        return field


class DepthEncoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 64,
        hidden_channels: int = 512,
        pos_enc_max_length: int = 1024,
        pos_enc_requires_grad: bool = False,
    ):
        super(DepthEncoder, self).__init__()

        self._positional_encoding = AbsolutePositionalEncoding(
            out_channels,
            max_length=pos_enc_max_length,
            requires_grad=pos_enc_requires_grad,
        )

        self._depth_projection = FeedForwardEncoder(1, out_channels, hidden_channels)

    def forward(self, layer_powers: torch.Tensor) -> torch.Tensor:

        # Accumulative summarize all powers to get depth
        layer_powers = torch.cumsum(layer_powers, dim=1).unsqueeze(-1)
        layer_powers = torch.log(layer_powers)

        # Encode depth
        layer_powers = self._depth_projection(layer_powers)

        # Add positional encoding
        layer_powers = self._positional_encoding(layer_powers)

        return layer_powers


if __name__ == "__main__":
    enc = FieldEncoder()
    enc(
        torch.randn((4, 1, 13, 32)),
        torch.Tensor([[0.01 * 2 ** i for i in range(13)] for _ in range(4)]),
    )
