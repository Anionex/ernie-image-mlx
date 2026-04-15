"""Self-Attention with QK-norm and RoPE for ERNIE-Image DiT."""
import mlx.core as mx
import mlx.nn as nn

from .embeddings import apply_rotary_emb


class Attention(nn.Module):
    """Multi-head self-attention with QK-norm and RoPE.

    Uses fused QKV projection and mx.fast.scaled_dot_product_attention.

    Weight loading: weights arrive as separate to_q/to_k/to_v keys.
    After loading, call fuse_qkv() to merge them into a single nn.Linear
    for ~4ms/block speedup.
    """

    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-6,
                 qk_norm: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size

        # Separate projections for weight loading compatibility
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)

        if qk_norm:
            self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
            self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)
        else:
            self.norm_q = None
            self.norm_k = None

        self.to_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self._qkv_fused = None

    def fuse_qkv(self):
        """Fuse Q/K/V weights into a single nn.Linear for faster matmul."""
        fused_weight = mx.concatenate([
            self.to_q.weight, self.to_k.weight, self.to_v.weight
        ], axis=0)
        self._qkv_fused = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=False)
        self._qkv_fused.load_weights([("weight", fused_weight)])

    def __call__(self, x: mx.array, mask: mx.array = None,
                 rotary_pos_emb: mx.array = None) -> mx.array:
        B, S, _ = x.shape

        if self._qkv_fused is not None:
            qkv = self._qkv_fused(x)
            q, k, v = mx.split(qkv, 3, axis=-1)
        else:
            q = self.to_q(x)
            k = self.to_k(x)
            v = self.to_v(x)

        q = q.reshape(B, S, self.num_heads, self.head_dim)
        k = k.reshape(B, S, self.num_heads, self.head_dim)
        v = v.reshape(B, S, self.num_heads, self.head_dim)

        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        if rotary_pos_emb is not None:
            q = apply_rotary_emb(q, rotary_pos_emb)
            k = apply_rotary_emb(k, rotary_pos_emb)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, mask=mask, scale=self.head_dim ** -0.5
        )

        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.to_out(out)
