import typing as tp

import torch
import torch.nn.functional as functional


# Given Query, Key, Value, calculate the final weighted value
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: tp.Optional[torch.Tensor] = None,
):
    # Shape of q and k are the same, both are (batch_size, seq_len, d_k)
    # Shape of v is (batch_size, seq_len, d_v)
    attention_scores = (
        torch.matmul(query, key.transpose(-2, -1)) / query.shape[-1] ** 0.5
    )  # size (batch_size, seq_len, seq_len)

    # Apply mask to scores
    # <pad>
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, value=-1e9)

    # Softmax along the last dimension
    attention_weights = functional.softmax(attention_scores, dim=-1)

    output = torch.matmul(attention_weights, value)
    return output


def linear_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: tp.Optional[torch.Tensor] = None,
):
    query = query.softmax(dim=-1)
    key = key.softmax(dim=-2)

    query = query * query.shape[0] ** -0.5

    context = torch.einsum("bnd,bne->bde", key, value)
    attn = torch.einsum("bnd,bde->bne", query, context)
    return attn.reshape(*query.shape)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads: int, hidden_channels: int, use_linear: bool = True):
        super().__init__()

        self.n_heads = num_heads
        self.d_model = hidden_channels
        self.d_k = hidden_channels // num_heads
        self.d_v = hidden_channels // num_heads

        # self attention linear layers
        # Linear layers for q, k, v vectors generation in different heads
        self.q_linear_layers = []
        self.k_linear_layers = []
        self.v_linear_layers = []
        for i in range(num_heads):
            self.q_linear_layers.append(torch.nn.Linear(hidden_channels, self.d_k))
            self.k_linear_layers.append(torch.nn.Linear(hidden_channels, self.d_k))
            self.v_linear_layers.append(torch.nn.Linear(hidden_channels, self.d_v))

        self.out = torch.nn.Linear(num_heads * self.d_v, hidden_channels)
        self.attn_function = (
            linear_dot_product_attention if use_linear else scaled_dot_product_attention
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: tp.Optional[torch.Tensor] = None,
    ):
        multi_head_attention_outputs = []
        for q_linear, k_linear, v_linear in zip(
            self.q_linear_layers, self.k_linear_layers, self.v_linear_layers
        ):
            new_q = q_linear(query)  # size: (batch_size, seq_len, d_k)
            new_k = k_linear(key)  # size: (batch_size, seq_len, d_k)
            new_v = v_linear(value)  # size: (batch_size, seq_len, d_v)

            # Scaled Dot-Product attention
            head_v = self.attn_function(new_q, new_k, new_v, mask)  # (batch_size, seq_len, d_v)
            multi_head_attention_outputs.append(head_v)

        # Concat
        # import pdb; pdb.set_trace()
        concat = torch.cat(multi_head_attention_outputs, -1)  # (batch_size, seq_len, n_heads*d_v)

        # Linear layer to recover to original shape
        output = self.out(concat)  # (batch_size, seq_len, d_model)

        return output
