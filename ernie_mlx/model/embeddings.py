"""Embedding modules: RoPE3D, TimestepEmbedding, PatchEmbed."""
import math

import mlx.core as mx
import mlx.nn as nn


def rope(pos: mx.array, dim: int, theta: int) -> mx.array:
    """Compute rotary position embedding frequencies.

    Args:
        pos: position indices [..., N]
        dim: embedding dimension (must be even)
        theta: base frequency
    Returns:
        frequencies [..., N, dim//2]
    """
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    omega = 1.0 / (theta ** scale)  # [dim//2]
    # einsum "...n,d->...nd"
    out = pos[..., None].astype(mx.float32) * omega  # [..., N, dim//2]
    return out


def rope_3d(ids: mx.array, axes_dim: tuple, theta: int) -> mx.array:
    """Compute 3D RoPE embeddings.

    Args:
        ids: [B, S, 3] position indices (temporal, height, width)
        axes_dim: dimensions for each axis, e.g. (32, 48, 48), sum = head_dim//2
        theta: base frequency
    Returns:
        [B, S, 1, head_dim] with pattern [θ0,θ0,θ1,θ1,...]
    """
    # Compute frequencies for each axis
    embs = []
    for i in range(3):
        embs.append(rope(ids[..., i], axes_dim[i], theta))
    emb = mx.concatenate(embs, axis=-1)  # [B, S, head_dim//2]
    emb = mx.expand_dims(emb, axis=2)    # [B, S, 1, head_dim//2]
    # Duplicate each frequency: [θ0,θ0,θ1,θ1,...]
    # stack on last dim then reshape
    emb = mx.stack([emb, emb], axis=-1)  # [B, S, 1, head_dim//2, 2]
    shape = list(emb.shape[:-2]) + [emb.shape[-2] * 2]
    emb = emb.reshape(*shape)            # [B, S, 1, head_dim]
    return emb


def apply_rotary_emb(x: mx.array, freqs: mx.array) -> mx.array:
    """Apply rotary position embedding (non-interleaved).

    Args:
        x: [B, S, heads, head_dim]
        freqs: [B, S, 1, head_dim] angle frequencies
    Returns:
        [B, S, heads, head_dim]
    """
    rot_dim = freqs.shape[-1]
    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]

    cos_ = mx.cos(freqs).astype(x.dtype)
    sin_ = mx.sin(freqs).astype(x.dtype)

    # Non-interleaved rotate_half: [-x2, x1]
    half = rot_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    x_rotated = mx.concatenate([-x2, x1], axis=-1)

    result = x_rot * cos_ + x_rotated * sin_
    if x_pass.shape[-1] > 0:
        result = mx.concatenate([result, x_pass], axis=-1)
    return result


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by 2-layer MLP."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


def timestep_sinusoidal(timesteps: mx.array, dim: int) -> mx.array:
    """Sinusoidal timestep encoding (matching diffusers Timesteps with flip_sin_to_cos=False).

    Args:
        timesteps: [B] tensor of timestep values
        dim: embedding dimension
    Returns:
        [B, dim] sinusoidal embedding
    """
    half_dim = dim // 2
    exponent = mx.arange(half_dim, dtype=mx.float32) / half_dim
    emb = mx.exp(-math.log(10000.0) * exponent)  # [half_dim]
    emb = timesteps[:, None].astype(mx.float32) * emb[None, :]  # [B, half_dim]
    # flip_sin_to_cos=False means [sin, cos] order
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)  # [B, dim]
    return emb


class PatchEmbed(nn.Module):
    """Patch embedding using Conv2d (patch_size=1 means just a linear projection)."""

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, H, W, C] in MLX NHWC format
        Returns:
            [B, H*W, hidden_size]
        """
        x = self.proj(x)  # [B, H', W', hidden_size]
        B, H, W, D = x.shape
        return x.reshape(B, H * W, D)
