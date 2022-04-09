import torch
import torch.nn as nn

from model.field_encoder import FieldEncoder, ResistivityEncoder
from model.transform import ResistivityTransform
from mtgrad.functional import direct_task

from linear_attention_transformer import LinearAttentionTransformer


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
        self,
        apparent_resistivity: torch.Tensor,
        impedance_phase: torch.Tensor,
        periods: torch.Tensor,
    ) -> torch.Tensor:
        apparent_resistivity = self._app_res_encoder(apparent_resistivity, periods)
        impedance_phase = self._imp_phs_encoder(impedance_phase, periods)
        return apparent_resistivity + impedance_phase


class MTFormer(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_decoder_blocks: int,
        num_attention_heads: int = 8,
        positional_encoding_requires_grad: bool = False,
        max_positional_length: int = 1024,
        max_time_steps: int = 100,
    ):
        super(MTFormer, self).__init__()

        self._rt = ResistivityTransform()

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
            max_time_steps=max_time_steps,
        )

        self._decoder = LinearAttentionTransformer(
            dim=hidden_channels,
            heads=num_attention_heads,
            depth=num_decoder_blocks,
            max_seq_len=max_time_steps,
            n_local_attn_heads=4,
            receives_context=True
        )

        self._out_projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_channels * 2, 1),
        )

    def forward(
        self,
        resistivity: torch.Tensor,
        time: torch.Tensor,
        apparent_resistivity: torch.Tensor,
        impedance_phase: torch.Tensor,
        periods: torch.Tensor,
        layer_powers: torch.Tensor,
    ) -> torch.Tensor:

        # 0. For shape preserving
        resistivity_shape = resistivity.shape

        # 1. Compute apparent resistivity & imp. phase
        app_res, imp_phs = direct_task(periods, resistivity, layer_powers)

        # 2. Compute log-resistivity and unsqueeze
        resistivity = self._rt.forward_transform(resistivity).unsqueeze(1)
        apparent_resistivity = self._rt.forward_transform(apparent_resistivity).unsqueeze(1)
        app_res = self._rt.forward_transform(app_res).unsqueeze(1)
        impedance_phase = impedance_phase.unsqueeze(1) / 50
        imp_phs = imp_phs.unsqueeze(1) / 50

        resistivity = resistivity * 1
        apparent_resistivity = apparent_resistivity * 1
        impedance_phase = impedance_phase * 1

        # 2. Encode resistivity & fields
        resistivity = self._base_res_encoder(resistivity, time)
        base_field = self._base_field_encoder(app_res, imp_phs, periods)
        target_field = self._target_field_encoder(apparent_resistivity, impedance_phase, periods)

        # 3. Reshape Fields & Resistivity to 1-d format
        target_field = target_field.view(target_field.shape[0], -1, target_field.shape[-1])
        base_field = base_field.view(base_field.shape[0], -1, base_field.shape[-1])
        resistivity = resistivity.view(resistivity.shape[0], -1, resistivity.shape[-1])

        # 5. Add fields together
        fields_encoded = target_field + base_field

        # 6. Encode sequence via transformer decoder blocks
        resistivity = self._decoder(resistivity, context=fields_encoded)

        # 7. Out projection & Reshape
        resistivity = self._out_projection(resistivity)
        resistivity = resistivity.view(*resistivity_shape)

        return resistivity
