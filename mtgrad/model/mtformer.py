import typing as tp

import torch
import torch.nn as nn

from mtgrad.model.field_encoder import DepthEncoder, FieldEncoder
from mtgrad.model.transform import FieldsTransform, ResistivityTransform


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
            hidden_channels,
            pos_enc_max_length,
            pos_enc_requires_grad,
            period_enc_log,
        )

        self._imp_phs_encoder = FieldEncoder(
            hidden_channels,
            pos_enc_max_length,
            pos_enc_requires_grad,
            period_enc_log,
        )

    def forward(
        self,
        apparent_resistivity: torch.Tensor,
        impedance_phase: torch.Tensor,
        periods: torch.Tensor,
    ) -> torch.Tensor:
        apparent_resistivity = self._app_res_encoder(
            apparent_resistivity, periods
        )
        impedance_phase = self._imp_phs_encoder(impedance_phase, periods)
        return torch.cat([apparent_resistivity, impedance_phase], dim=1)


class MTFormer(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_decoder_blocks: int,
        num_attention_heads: int = 8,
        positional_encoding_requires_grad: bool = False,
        max_positional_length: int = 128,
        quantization: tp.Optional[int] = None,
    ):
        super(MTFormer, self).__init__()

        self._rt = ResistivityTransform()
        self._ft = FieldsTransform()

        self._out_channels = quantization if quantization is not None else 1

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

        self._depth_encoder = DepthEncoder(
            hidden_channels,
            pos_enc_requires_grad=positional_encoding_requires_grad,
            pos_enc_max_length=max_positional_length,
        )

        self._decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=hidden_channels,
                nhead=num_attention_heads,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_decoder_blocks,
        )

        self._out_projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_channels * 2, self._out_channels),
        )

    def normalize_resistivity(self, resistivity: torch.Tensor) -> torch.Tensor:
        return self._rt.normalize(resistivity)

    def denormalize_resistivity(
        self, resistivity: torch.Tensor
    ) -> torch.Tensor:
        return self._rt.denormalize(resistivity)

    def forward(
        self,
        apparent_resistivity: torch.Tensor,
        impedance_phase: torch.Tensor,
        periods: torch.Tensor,
        layer_powers: torch.Tensor,
    ) -> torch.Tensor:

        # 1. Acquire resistivity depth grid
        resistivity_grid = self._depth_encoder(layer_powers)
        resistivity_grid_shape = resistivity_grid.shape[:-1]

        # 2. Transform fields
        apparent_resistivity = self._ft.normalize_app_res(
            apparent_resistivity
        ).unsqueeze(1)
        impedance_phase = self._ft.normalize_imp_phs(impedance_phase).unsqueeze(
            1
        )

        # 2. Encode input fields
        target_field = self._target_field_encoder(
            apparent_resistivity, impedance_phase, periods
        ).contiguous()

        # 3. Reshape Fields & Resistivity to 1-d format
        target_field = target_field.view(
            target_field.shape[0], -1, target_field.shape[-1]
        )
        resistivity_grid = resistivity_grid.view(
            resistivity_grid.shape[0], -1, resistivity_grid.shape[-1]
        )

        # 4. Encode sequence via transformer decoder blocks
        resistivity_grid = self._decoder(resistivity_grid, target_field)

        # 5. Out projection & Reshape
        resistivity_grid = self._out_projection(resistivity_grid)
        resistivity_grid = resistivity_grid.view(*resistivity_grid_shape)

        return resistivity_grid
