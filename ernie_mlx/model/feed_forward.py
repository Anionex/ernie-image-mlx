"""Gated Feed-Forward Network for ERNIE-Image DiT."""
import mlx.core as mx
import mlx.nn as nn


class FeedForward(nn.Module):
    """Gated FFN: down(up(x) * gelu(gate(x)))

    Supports gate+up fusion: after weight loading, call fuse_gate_up()
    to merge gate_proj and up_proj into a single nn.Linear for ~4ms savings.
    """

    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)
        self._gate_up_fused = None

    def fuse_gate_up(self):
        """Fuse gate+up weights into a single nn.Linear for faster matmul."""
        fused_weight = mx.concatenate([
            self.gate_proj.weight, self.up_proj.weight
        ], axis=0)
        self._gate_up_fused = nn.Linear(self.hidden_size, self.ffn_hidden_size * 2, bias=False)
        self._gate_up_fused.load_weights([("weight", fused_weight)])

    def __call__(self, x: mx.array) -> mx.array:
        if self._gate_up_fused is not None:
            gate_up = self._gate_up_fused(x)
            gate, up = mx.split(gate_up, 2, axis=-1)
            return self.linear_fc2(up * nn.gelu(gate))
        return self.linear_fc2(self.up_proj(x) * nn.gelu(self.gate_proj(x)))
