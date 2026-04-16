"""Single DiT block with shared AdaLN modulation."""
import mlx.core as mx
import mlx.nn as nn

from .attention import Attention
from .feed_forward import FeedForward


class DiTBlock(nn.Module):
    """ErnieImage shared AdaLN transformer block.

    Pattern:
        residual + gate_msa * attn(adaln(norm(x), shift_msa, scale_msa))
        residual + gate_mlp * ffn(adaln(norm(x), shift_mlp, scale_mlp))

    AdaLN intermediate computations use float32 for numerical stability.
    """

    def __init__(self, hidden_size: int, num_heads: int, ffn_hidden_size: int,
                 eps: float = 1e-6, qk_norm: bool = True):
        super().__init__()
        self.adaLN_sa_ln = nn.RMSNorm(hidden_size, eps=eps)
        self.self_attention = Attention(hidden_size, num_heads, eps=eps, qk_norm=qk_norm)
        self.adaLN_mlp_ln = nn.RMSNorm(hidden_size, eps=eps)
        self.mlp = FeedForward(hidden_size, ffn_hidden_size)

    def __call__(self, x: mx.array, rotary_pos_emb: mx.array,
                 temb: tuple, attention_mask: mx.array = None) -> mx.array:
        """
        Args:
            x: [B, S, H] hidden states (batch-first in MLX)
            rotary_pos_emb: [B, S, 1, head_dim] RoPE frequencies
            temb: tuple of 6 tensors (shift_msa, scale_msa, gate_msa,
                   shift_mlp, scale_mlp, gate_mlp), each [B, 1, H]
            attention_mask: [B, 1, 1, S] boolean mask
        Returns:
            [B, S, H]
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = temb

        # Self-attention branch
        residual = x
        x_norm = self.adaLN_sa_ln(x)
        x_mod = x_norm * (1 + scale_msa) + shift_msa
        attn_out = self.self_attention(x_mod, mask=attention_mask, rotary_pos_emb=rotary_pos_emb)
        x = residual + gate_msa * attn_out

        # FFN branch
        residual = x
        x_norm = self.adaLN_mlp_ln(x)
        x_mod = x_norm * (1 + scale_mlp) + shift_mlp
        ffn_out = self.mlp(x_mod)
        x = residual + gate_mlp * ffn_out

        return x
