import torch
import torch.nn as nn

from data.dataset import RandomLayerDataset
from model.field_encoder import FieldEncoder, ResistivityEncoder
from model.transformer import ResistivityDecoderLayer


class MTFieldEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        pos_enc_max_length: int = 1024,
        pos_enc_requires_grad: bool = False,
        period_enc_log: bool = True,
    ):
        super(MTFieldEncoder, self).__init__()

        self._app_res_encoder = FieldEncoder(
            hidden_channels, pos_enc_max_length, pos_enc_requires_grad, period_enc_log
        )

        self._imp_phs_encoder = FieldEncoder(
            hidden_channels, pos_enc_max_length, pos_enc_requires_grad, period_enc_log
        )

    def forward(
        self, apparent_resistivity: torch.Tensor, impedance_phase: torch.Tensor, periods: torch.Tensor
    ) -> torch.Tensor:
        apparent_resistivity = self._app_res_encoder(apparent_resistivity, periods)
        impedance_phase = self._imp_phs_encoder(impedance_phase, periods)
        return apparent_resistivity + impedance_phase


class MTFormer(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_decoder_blocks: int,
        use_linear_attention: bool = False,
        num_attention_heads: int = 8,
        positional_encoding_requires_grad: bool = False,
        max_positional_length: int = 1024,
    ):
        super(MTFormer, self).__init__()

        self._base_field_encoder = MTFieldEncoder(
            hidden_channels,
            pos_enc_requires_grad=positional_encoding_requires_grad,
            pos_enc_max_length=max_positional_length,
        )

        self._target_field_encoder = MTFieldEncoder(
            hidden_channels,
            pos_enc_requires_grad=positional_encoding_requires_grad,
            pos_enc_max_length=max_positional_length,
        )

        self._base_res_encoder = ResistivityEncoder(
            hidden_channels,
            pos_enc_requires_grad=positional_encoding_requires_grad,
            pos_enc_max_length=max_positional_length,
        )

        self._decoder_blocks = nn.ModuleList()
        for _ in range(num_decoder_blocks):
            decoder_block = ResistivityDecoderLayer(
                hidden_channels, num_attention_heads, use_linear_attn=use_linear_attention
            )
            self._decoder_blocks.append(decoder_block)

        self._out_projection = nn.Linear(hidden_channels, 1)

    def forward(
        self,
        base_resistivity: torch.Tensor,
        base_apparent_resistivity: torch.Tensor,
        base_impedance_phase: torch.Tensor,
        target_apparent_resistivity: torch.Tensor,
        target_impedance_phase: torch.Tensor,
        periods: torch.Tensor,
    ) -> torch.Tensor:

        resistivity = self._base_res_encoder(base_resistivity)
        base_field = self._base_field_encoder(base_apparent_resistivity, base_impedance_phase, periods)
        target_field = self._target_field_encoder(
            target_apparent_resistivity, target_impedance_phase, periods
        )

        target_field = target_field.view(target_field.shape[0], -1, target_field.shape[-1])
        base_field = base_field.view(base_field.shape[0], -1, base_field.shape[-1])
        resistivity = resistivity.view(resistivity.shape[0], -1, resistivity.shape[-1])

        for block in self._decoder_blocks:
            resistivity = block(resistivity, base_field, target_field)
        resistivity = self._out_projection(resistivity)
        resistivity = resistivity.view(*base_resistivity.shape)
        return resistivity
