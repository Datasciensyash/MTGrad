import math
import random

import torch
import torch.nn as nn

from model.modules import FeedForwardEncoder


class AbsolutePositionalEncoding(nn.Module):
    # TODO: Maybe we should use Relative embeddings?
    def __init__(self, in_channels: int, max_length: int = 1024, requires_grad: bool = False):
        super(AbsolutePositionalEncoding, self).__init__()
        self._in_channels = in_channels
        self._max_length = max_length

        if requires_grad:
            positional_encoding = self.initialize_random_encoding_table()
        else:
            positional_encoding = self.initialize_sinusoid_encoding_table()

        self._positional_encoding = nn.Parameter(positional_encoding, requires_grad=requires_grad)
        self._alpha = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

        self.register_parameter("_positional_encoding", self._positional_encoding)
        self.register_parameter("_alpha", self._alpha)

    def initialize_sinusoid_encoding_table(self) -> torch.Tensor:
        positional_encoding = torch.empty((self._max_length, self._in_channels))
        for pos_index in range(self._max_length):
            for i in range(0, self._in_channels, 2):
                sin_input = pos_index / 1.0e4 ** ((2 * i) / self._in_channels)
                cos_input = pos_index / 1.0e4 ** ((2 * i + 1) / self._in_channels)
                positional_encoding[pos_index, i] = math.sin(sin_input)
                positional_encoding[pos_index, i + 1] = math.cos(cos_input)
        return positional_encoding

    def initialize_random_encoding_table(self) -> torch.Tensor:
        positional_encoding = torch.empty((self._max_length, self._in_channels))
        nn.init.kaiming_uniform_(positional_encoding, mode="fan_in")
        return positional_encoding

    def forward(self, input_tensor: torch.Tensor, random_slice: bool = False) -> torch.Tensor:
        tensor_length = input_tensor.shape[2]  # (B, H, W, C) -> W
        start_index = 0 if not random_slice else random.randint(0, self._max_length - tensor_length)
        encoding = self._positional_encoding[start_index : start_index + tensor_length]
        # (B, H, W, C) + (W, C) -> (B, H, W, C)
        return self._alpha * encoding + input_tensor


class PeriodEncoder(nn.Module):
    def __init__(self, out_channels: int, log: bool = True):
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
        pos_enc_requires_grad: bool = False,
        period_enc_log: bool = True,
    ):
        super(FieldEncoder, self).__init__()

        self._positional_encoding = AbsolutePositionalEncoding(
            hidden_channels,
            max_length=pos_enc_max_length,
            requires_grad=pos_enc_requires_grad,
        )

        self._period_encoding = PeriodEncoder(hidden_channels, log=period_enc_log)

        self._field_projection = FeedForwardEncoder(1, hidden_channels)

    def forward(self, field: torch.Tensor, periods: torch.Tensor) -> torch.Tensor:
        # Encode field
        field = torch.einsum("b c h w -> b h w c", field)
        field = self._field_projection(field)

        # Add positional encoding
        field = self._positional_encoding(field)

        # Add period encoding
        field = torch.einsum("b h w c, b h c -> b h w c", field, self._period_encoding(periods))

        return field


class ResistivityEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        pos_enc_max_length: int = 1024,
        pos_enc_requires_grad: bool = False,
        max_time_steps: int = 100,
    ):
        super(ResistivityEncoder, self).__init__()

        self._positional_encoding = AbsolutePositionalEncoding(
            hidden_channels,
            max_length=pos_enc_max_length,
            requires_grad=pos_enc_requires_grad,
        )

        self._time_encoding = nn.Embedding(max_time_steps, hidden_channels)

        self._res_projection = FeedForwardEncoder(1, hidden_channels)

    def forward(self, resistivity: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # Encode resistivity
        resistivity = torch.einsum("b c h w -> b h w c", resistivity)
        resistivity = self._res_projection(resistivity)

        # Add positional encoding
        resistivity = self._positional_encoding(resistivity)

        # Add time encoding
        time_encoded = self._time_encoding(time)
        resistivity = torch.einsum("b w h c, b c -> b w h c", resistivity, time_encoded)

        return resistivity


if __name__ == "__main__":
    enc = FieldEncoder()
    enc(
        torch.randn((4, 1, 13, 32)),
        torch.Tensor([[0.01 * 2 ** i for i in range(13)] for _ in range(4)]),
    )
