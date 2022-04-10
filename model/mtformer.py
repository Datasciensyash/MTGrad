import torch
import torch.nn as nn
from routing_transformer import RoutingTransformer

from model.field_encoder import FieldEncoder, ResistivityEncoder
from model.transform import ResistivityTransform
from mtgrad.functional import direct_task
from torch.nn import functional

from linear_attention_transformer import LinearAttentionTransformer, LinformerContextSettings

from performer_pytorch import Performer, PerformerEncDec, CrossAttention
from memformer import Memformer


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

        self.__decoder = RoutingTransformer(
            dim=hidden_channels,
            heads=num_attention_heads,
            depth=num_decoder_blocks,
            max_seq_len=1024,
            window_size=128,
            n_local_attn_heads=4,
            receives_context=True,
            local_attn_window_size=64
        )

        self.__decoder = Performer(
            dim=hidden_channels,
            depth=num_decoder_blocks,
            heads=num_attention_heads,
            cross_attend=True,
            dim_head=hidden_channels,
            nb_features=hidden_channels
        )

        self._att = CrossAttention(
            dim=hidden_channels,
            dim_head=hidden_channels,
            heads=num_attention_heads,
        )

        self._decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=hidden_channels,
                nhead=num_attention_heads,
                batch_first=True
            ),
            num_layers=num_decoder_blocks
        )

        self.__decoder = LinearAttentionTransformer(
            dim=hidden_channels,
            heads=num_attention_heads,
            depth=num_decoder_blocks,
            max_seq_len=max_time_steps,
            n_local_attn_heads=4,
            receives_context=True,
            context_linformer_settings=None,
            linformer_settings=None
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
        # resistivity = resistivity.abs()

        # 0. For shape preserving
        resistivity_shape = resistivity.shape

        # 1. Compute apparent resistivity & imp. phase
        #app_res, imp_phs = direct_task(periods, self._rt.backward_transform(resistivity), layer_powers)

        # 2. Compute log-resistivity and unsqueeze
        resistivity_grad = resistivity.unsqueeze(1)
        apparent_resistivity = self._rt.forward_transform(apparent_resistivity).unsqueeze(1)
        #app_res = self._rt.forward_transform(app_res).unsqueeze(1)
        impedance_phase = impedance_phase.unsqueeze(1) / 50
        #imp_phs = imp_phs.unsqueeze(1) / 50

        # 2. Encode resistivity & fields
        resistivity_grad = self._base_res_encoder(resistivity_grad, time)
        #base_field = self._base_field_encoder(app_res, imp_phs, periods)
        target_field = self._target_field_encoder(apparent_resistivity, impedance_phase, periods)

        # 3. Reshape Fields & Resistivity to 1-d format
        target_field = target_field.view(target_field.shape[0], -1, target_field.shape[-1])
        #base_field = base_field.view(base_field.shape[0], -1, base_field.shape[-1])
        resistivity_grad = resistivity_grad.view(resistivity_grad.shape[0], -1, resistivity_grad.shape[-1])

        # 5. Add fields together
        fields_encoded = target_field #+ base_field

        # 6. Encode sequence via transformer decoder blocks
        resistivity_grad = self._decoder(resistivity_grad, fields_encoded)

        # 7. Out projection & Reshape
        resistivity_grad = self._out_projection(resistivity_grad)
        resistivity_grad = resistivity_grad.view(*resistivity_shape)

        return resistivity_grad
