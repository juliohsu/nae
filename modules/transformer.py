"""A streamable transformer."""

import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_sin_embedding(positions: torch.Tensor, dim: int, max_period: float = 10000):
    """Create time embedding for the given position, target dimension `dim`."""
    # we aim for BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    adim = torch.arange(half_dim, device=positions.device).view(1, 1, -1)
    phase = positions / max_period ** (adim / (half_dim - 1))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=1)


class StreamingTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(
        self, x: torch.Tensor, x_past: torch.Tensor, past_context: int
    ):  # type ignore
        if self.norm_first:
            sa_input = self.norm1(x)
            x = x + self._sa_block(sa_input, x_past, past_context)
            x = x + self._ff_block(x)
        else:
            sa_input = x
            x = self.norm1(x + self._sa_block(sa_input, x_past, past_context))
            x = self.norm2(x + self._ff_block(x))
        return x, sa_input

    # self-attention block
    def _sa_block(
        self, x: torch.Tensor, x_past: torch.Tensor, past_context: int
    ):  # type ignore
        _, T, _ = x.shape
        _, H, _ = x_past.shape

        queries = x
        keys = torch.cat([x, x_past], dim=1)
        values = keys

        queries_pos = torch.arange(H, H + T, device=x.device).view(-1, 1)
        keys_pos = torch.arange(H + T, device=x.device).view(1, -1)
        delta = queries_pos - keys_pos
        valid_access = (delta >= 0) & (delta <= past_context)

        x = self.self_attn(
            queries, keys, values, attn_ask=~valid_access, need_weights=False
        )[0]

        return self.dropout1(x)


class StreamingTransformerEncoder(nn.Module):
    """Transformer encoder with streaming support.
    Args:
        dim,
        num_heads,
        num_layers,
        max_period,
        past_context,
        gelu,
        norm_in,
        dropout,
        **kwargs
    """

    def __init__(
        self,
        dim,
        hidden_scale: float = 4.0,
        num_heads: int = 8,
        num_layers: int = 5,
        max_period: int = 10000,
        past_context: int = 1000,
        gelu: bool = True,
        norm_in: bool = True,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        assert dim % num_heads == 0
        hidden_dim = int(dim * hidden_scale)

        self.max_period = max_period
        self.past_context = past_context
        activation: tp.Any = F.gelu if gelu else F.relu

        self.norm_in: nn.Module
        if norm_in:
            self.norm_in = nn.LayerNorm(dim)
        else:
            self.norm_in = nn.Identity()

        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            self.layers.append(
                StreamingTransformerEncoderLayer(
                    dim,
                    num_heads,
                    hidden_dim,
                    activation=activation,
                    batch_first=True,
                    dropout=dropout,
                    **kwargs
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        states: tp.Optional[tp.List[torch.Tensor]] = None,
        offset: tp.Union[int, torch.Tensor] = 0,
    ):
        B, T, C = x.shape
        positions = torch.arange(T, device=x.device).view(1, -1, 1) + offset
        pos_emb = create_sin_embedding(positions, C, max_period=self.max_period)

        x = self.norm_in(x)
        x += pos_emb

        if states is None:
            states = [torch.zeros_like(x[:, :1]) for _ in range(self.layers + 1)]
        new_states: tp.List[torch.Tensor] = []

        for state, layer in zip(states, self.layers):
            x, new_state_layer = layer(x, state, self.past_context)
            new_state_layer = torch.cat([state, new_state_layer], dim=1)
            new_states.append(new_state_layer[:, -self.past_context :, :])
        return x, new_states, offset + T
