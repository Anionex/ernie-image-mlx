"""Mistral3 text encoder for ERNIE-Image (MLX).

26-layer GQA Transformer with YaRN RoPE. Only forward pass needed
(no KV cache, no generation). Returns hidden_states[-2] for the DiT.
"""
import math

import mlx.core as mx
import mlx.nn as nn

from ..config import TextEncoderConfig


def _compute_yarn_inv_freq(config: TextEncoderConfig) -> mx.array:
    """Compute YaRN-scaled inverse frequencies for RoPE.

    Implements the YaRN (Yet another RoPE extensioN) paper's frequency scaling
    to extend the context window from original_max to max_position_embeddings.
    """
    dim = config.head_dim
    base = config.rope_theta
    factor = config.rope_scaling_factor
    original_max = config.rope_original_max_pos
    beta_fast = config.yarn_beta_fast
    beta_slow = config.yarn_beta_slow

    # Base inverse frequencies (no scaling)
    freq_extra = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    # Interpolated frequencies (scaled by factor)
    freq_inter = freq_extra / factor

    # Find correction dimensions
    def find_correction_dim(num_rotations: float) -> float:
        return (dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    low = math.floor(find_correction_dim(beta_fast))
    high = math.ceil(find_correction_dim(beta_slow))

    # Linear ramp between low and high
    if low == high:
        high = low + 1
    t = mx.arange(dim // 2, dtype=mx.float32)
    ramp = mx.clip((t - low) / (high - low), 0.0, 1.0)

    # Blend: ramp=0 means extrapolation (original freq), ramp=1 means interpolation
    inv_freq = freq_inter * ramp + freq_extra * (1 - ramp)
    return inv_freq


class MistralRoPE(nn.Module):
    """YaRN Rotary Position Embedding."""

    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self._inv_freq = _compute_yarn_inv_freq(config)

    def __call__(self, x: mx.array, position_ids: mx.array):
        """
        Args:
            x: unused, just for device/dtype reference
            position_ids: [B, S] position indices
        Returns:
            (cos, sin): each [B, S, head_dim]
        """
        # [B, S, 1] * [1, 1, dim//2] -> [B, S, dim//2]
        freqs = position_ids[:, :, None].astype(mx.float32) * self._inv_freq[None, None, :]
        # Duplicate for rotate_half: [B, S, dim]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


def rotate_half(x: mx.array) -> mx.array:
    """Non-interleaved rotate_half: [x1, x2] -> [-x2, x1]."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array):
    """Apply rotary embeddings to q and k.

    Args:
        q: [B, num_heads, S, head_dim]
        k: [B, num_kv_heads, S, head_dim]
        cos, sin: [B, S, head_dim] -> unsqueeze to [B, 1, S, head_dim]
    """
    cos = cos[:, None, :, :]  # [B, 1, S, head_dim]
    sin = sin[:, None, :, :]
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class MistralAttention(nn.Module):
    """Group Query Attention (GQA) for Mistral."""

    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array,
                 mask: mx.array = None) -> mx.array:
        """
        Args:
            x: [B, S, hidden_size]
            cos, sin: [B, S, head_dim] from RoPE
            mask: [B, 1, S, S] causal mask (additive, 0=attend, -inf=mask)
        Returns:
            [B, S, hidden_size]
        """
        B, S, _ = x.shape

        q = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: repeat KV heads
        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=1)
            v = mx.repeat(v, self.num_kv_groups, axis=1)

        # Scaled dot-product attention
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, mask=mask, scale=self.head_dim ** -0.5
        )

        # [B, heads, S, head_dim] -> [B, S, heads*head_dim]
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out)


class MistralMLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MistralDecoderLayer(nn.Module):
    """Single Mistral decoder layer."""

    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.self_attn = MistralAttention(config)
        self.mlp = MistralMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array,
                 mask: mx.array = None) -> mx.array:
        # Pre-norm + attention + residual
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin, mask)
        x = residual + x
        # Pre-norm + FFN + residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class MistralTextEncoder(nn.Module):
    """Mistral3 text encoder (language model only, no vision tower).

    Forward returns hidden_states[-2]: output of layer N-2 (before last layer
    and final norm), matching the diffusers pipeline behavior.
    """

    def __init__(self, config: TextEncoderConfig = TextEncoderConfig()):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [MistralDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MistralRoPE(config)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array = None) -> mx.array:
        """
        Args:
            input_ids: [B, S] token IDs
            attention_mask: [B, S] boolean mask (True=valid, False=padding)
        Returns:
            [B, S, hidden_size] hidden states from second-to-last layer
        """
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids)

        # Position IDs: simple arange
        position_ids = mx.broadcast_to(mx.arange(S).reshape(1, S), (B, S))
        cos, sin = self.rotary_emb(x, position_ids)

        # Causal mask + padding mask
        # Create causal mask: [1, 1, S, S]
        causal = mx.triu(mx.full((S, S), -1e9), k=1)
        causal = causal.reshape(1, 1, S, S)

        if attention_mask is not None:
            # Padding mask: [B, 1, 1, S] — mask out padding positions
            pad_mask = mx.where(attention_mask[:, None, None, :], 0.0, -1e9)
            mask = causal + pad_mask
        else:
            mask = causal

        # Forward through layers, collect hidden states
        # hidden_states[0] = embedding, hidden_states[i+1] = output of layer i
        # We need hidden_states[-2] = output of layer (N-2), i.e., index N-1 in 0-based
        target_layer = self.config.num_hidden_layers - 2  # layer index to return

        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, mask)
            if i == target_layer:
                output = x  # Save second-to-last layer output

        return output
