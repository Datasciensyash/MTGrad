import torch
import torch.nn as nn
import torch.nn.functional as functional
import typing as tp
from model.attention import MultiHeadAttention


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        self.linear_1 = torch.nn.Linear(d_model, d_ff)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear_2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(functional.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class LayerNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.alpha = torch.nn.Parameter(torch.ones(self.d_model))
        self.beta = torch.nn.Parameter(torch.zeros(self.d_model))
        self.eps = eps

    def forward(self, x):
        # x size: (batch_size, seq_len, d_model)
        x_hat = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps)
        x_tilde = self.alpha * x_hat + self.beta
        return x_tilde


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention(n_heads, d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout_1 = torch.nn.Dropout(dropout)
        self.dropout_2 = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        # import pdb; pdb.set_trace()
        x = x + self.dropout_1(self.multi_head_attention(x, x, x, mask))
        x = self.norm_1(x)

        x = x + self.dropout_2(self.feed_forward(x))
        x = self.norm_2(x)
        return x


class ResistivityDecoderLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        dropout_ratio: float = 0.1,
        use_linear_attn: bool = False,
    ):
        super().__init__()
        self.norm_1 = LayerNorm(hidden_channels)
        self.norm_2 = LayerNorm(hidden_channels)
        self.norm_3 = LayerNorm(hidden_channels)
        self.norm_4 = LayerNorm(hidden_channels)

        self.dropout_1 = torch.nn.Dropout(dropout_ratio)
        self.dropout_2 = torch.nn.Dropout(dropout_ratio)
        self.dropout_3 = torch.nn.Dropout(dropout_ratio)
        self.dropout_4 = torch.nn.Dropout(dropout_ratio)

        self.multi_head_attention_1 = MultiHeadAttention(
            num_heads, hidden_channels, use_linear_attn
        )
        self.multi_head_attention_2 = MultiHeadAttention(
            num_heads, hidden_channels, use_linear_attn
        )
        self.multi_head_attention_3 = MultiHeadAttention(
            num_heads, hidden_channels, use_linear_attn
        )

        self.feed_forward = FeedForward(hidden_channels)

    def forward(
        self,
        resistivity: torch.Tensor,
        base_field_encoded: torch.Tensor,
        target_field_encoded: torch.Tensor,
        src_mask: tp.Optional[torch.Tensor] = None,
        trg_mask: tp.Optional[torch.Tensor] = None,
    ):

        # Resistivity self-attention
        resistivity = self.dropout_1(
            self.multi_head_attention_1(resistivity, resistivity, resistivity, trg_mask)
        )
        resistivity = resistivity + self.norm_1(resistivity)

        # Target-to-base field attention
        target_field_encoded = self.dropout_2(
            self.multi_head_attention_2(
                target_field_encoded, base_field_encoded, base_field_encoded, trg_mask
            )
        )
        target_field_encoded = target_field_encoded + self.norm_2(target_field_encoded)

        # Resistivity cross-attention to target field
        resistivity = self.dropout_3(
            self.multi_head_attention_3(
                resistivity, target_field_encoded, target_field_encoded, src_mask
            )
        )
        resistivity = resistivity + self.norm_3(resistivity)

        resistivity = self.dropout_4(self.feed_forward(resistivity))
        resistivity = resistivity + self.norm_4(resistivity)

        return resistivity


if __name__ == "__main__":
    l = ResistivityDecoderLayer(512, 8)
    output_resistivity = torch.randn((1, 8 * 32, 512))
    output_field = torch.randn((1, 13 * 32, 512))
    a = l(output_resistivity, output_field, None, None)
    print(a.shape)
