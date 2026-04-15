"""ERNIE-Image DiT Transformer model."""
import mlx.core as mx
import mlx.nn as nn

from ..config import TransformerConfig
from .dit_block import DiTBlock
from .embeddings import PatchEmbed, TimestepEmbedding, rope_3d, timestep_sinusoidal


class AdaLNContinuous(nn.Module):
    """Final adaptive layer norm with continuous conditioning."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, affine=False, eps=eps)
        self.linear = nn.Linear(hidden_size, hidden_size * 2)

    def __call__(self, x: mx.array, conditioning: mx.array) -> mx.array:
        """
        Args:
            x: [B, S, H]
            conditioning: [B, H]
        Returns:
            [B, S, H]
        """
        scale, shift = mx.split(self.linear(conditioning), 2, axis=-1)
        x = self.norm(x)
        # Broadcast conditioning [B, 1, H] to sequence dim
        x = x * (1 + scale[:, None, :]) + shift[:, None, :]
        return x


class ErnieImageTransformer(nn.Module):
    """Full ERNIE-Image DiT model.

    Architecture:
        1. Patch embed image tokens + project text tokens
        2. Concatenate [image, text] tokens
        3. Compute 3D RoPE position embeddings
        4. Compute timestep conditioning (shared AdaLN)
        5. Run through 36 DiT blocks
        6. Final norm + linear projection
        7. Extract image tokens and unpatchify
    """

    def __init__(self, config: TransformerConfig = TransformerConfig()):
        super().__init__()
        self.config = config
        h = config.hidden_size

        self.x_embedder = PatchEmbed(config.in_channels, h, config.patch_size)

        if config.text_in_dim != h:
            self.text_proj = nn.Linear(config.text_in_dim, h, bias=False)
        else:
            self.text_proj = None

        self.time_embedding = TimestepEmbedding(h)

        # Shared AdaLN modulation: SiLU + Linear -> 6 * hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(h, 6 * h)
        )

        self.layers = [
            DiTBlock(h, config.num_attention_heads, config.ffn_hidden_size,
                     config.eps, config.qk_layernorm)
            for _ in range(config.num_layers)
        ]

        self.final_norm = AdaLNContinuous(h, config.eps)
        self.final_linear = nn.Linear(h, config.patch_size ** 2 * config.out_channels)

        # RoPE params (not learnable)
        self.rope_theta = config.rope_theta
        self.rope_axes_dim = config.rope_axes_dim

    def fuse_qkv_weights(self):
        """Fuse Q/K/V projections in all attention layers."""
        for layer in self.layers:
            layer.self_attention.fuse_qkv()

    def fuse_ffn_weights(self):
        """Fuse gate+up projections in all FFN layers."""
        for layer in self.layers:
            layer.mlp.fuse_gate_up()

    def prepare_inputs(self, B: int, H_img: int, W_img: int,
                       text_bth: mx.array, text_lens: mx.array):
        """Precompute values that stay constant across denoising steps.

        Returns:
            dict with 'text_proj', 'rotary_pos_emb', 'attention_mask', 'N_img'
        """
        N_img = H_img * W_img
        Tmax = text_bth.shape[1]

        # Project text (constant across steps)
        if self.text_proj is not None and text_bth.size > 0:
            text_bth = self.text_proj(text_bth)

        # Position IDs for 3D RoPE
        row_ids = mx.repeat(mx.arange(H_img, dtype=mx.float32), W_img)
        col_ids = mx.tile(mx.arange(W_img, dtype=mx.float32), (H_img,))
        grid_yx = mx.stack([row_ids, col_ids], axis=-1)

        text_offset = text_lens.astype(mx.float32).reshape(B, 1, 1)
        text_offset = mx.broadcast_to(text_offset, (B, N_img, 1))
        grid_yx_expanded = mx.broadcast_to(grid_yx.reshape(1, N_img, 2), (B, N_img, 2))
        image_ids = mx.concatenate([text_offset, grid_yx_expanded], axis=-1)

        if Tmax > 0:
            text_seq = mx.broadcast_to(
                mx.arange(Tmax, dtype=mx.float32).reshape(1, Tmax, 1), (B, Tmax, 1)
            )
            text_zeros = mx.zeros((B, Tmax, 2))
            text_ids = mx.concatenate([text_seq, text_zeros], axis=-1)
        else:
            text_ids = mx.zeros((B, 0, 3))

        all_ids = mx.concatenate([image_ids, text_ids], axis=1)
        rotary_pos_emb = rope_3d(all_ids, self.rope_axes_dim, self.rope_theta)

        # Attention mask — skip if all text tokens are valid (no padding)
        # This saves ~6.5ms/layer = 234ms/step by letting SDPA use the faster no-mask path
        S = N_img + Tmax
        all_valid = Tmax == 0 or bool(mx.all(text_lens >= Tmax).item())
        if all_valid:
            attention_mask = None
        else:
            text_range = mx.arange(Tmax).reshape(1, Tmax)
            valid_text = text_range < text_lens.reshape(B, 1)
            img_mask = mx.ones((B, N_img), dtype=mx.bool_)
            attention_mask = mx.concatenate([img_mask, valid_text], axis=1)
            attention_mask = attention_mask.reshape(B, 1, 1, S)

        return {
            'text_projected': text_bth,
            'rotary_pos_emb': rotary_pos_emb,
            'attention_mask': attention_mask,
            'N_img': N_img,
        }

    def __call__(self, hidden_states: mx.array, timestep: mx.array,
                 text_bth: mx.array, text_lens: mx.array,
                 cached: dict = None) -> mx.array:
        """
        Args:
            hidden_states: [B, H, W, C] latent image in NHWC format
            timestep: [B] timestep values
            text_bth: [B, T, text_dim] text hidden states
            text_lens: [B] valid text lengths
            cached: precomputed values from prepare_inputs (optional)
        Returns:
            [B, H, W, C] denoised output in NHWC
        """
        B = hidden_states.shape[0]
        H_img = hidden_states.shape[1] // self.config.patch_size
        W_img = hidden_states.shape[2] // self.config.patch_size
        p = self.config.patch_size

        # 1. Patch embed
        img_tokens = self.x_embedder(hidden_states)

        if cached is not None:
            text_projected = cached['text_projected']
            rotary_pos_emb = cached['rotary_pos_emb']
            attention_mask = cached['attention_mask']
            N_img = cached['N_img']
        else:
            # Fallback: compute everything inline
            prep = self.prepare_inputs(B, H_img, W_img, text_bth, text_lens)
            text_projected = prep['text_projected']
            rotary_pos_emb = prep['rotary_pos_emb']
            attention_mask = prep['attention_mask']
            N_img = prep['N_img']

        # 2. Concatenate [image, text]
        x = mx.concatenate([img_tokens, text_projected], axis=1)

        # 3. Timestep conditioning (changes every step)
        t_emb = timestep_sinusoidal(timestep, self.config.hidden_size)
        c = self.time_embedding(t_emb)
        ada_out = self.adaLN_modulation(c)
        ada_chunks = mx.split(ada_out, 6, axis=-1)
        temb = tuple(chunk[:, None, :] for chunk in ada_chunks)

        # 4. Transformer layers
        for layer in self.layers:
            x = layer(x, rotary_pos_emb, temb, attention_mask)

        # 5. Final norm + projection
        x = self.final_norm(x, c)
        x = self.final_linear(x)

        # 6. Extract image tokens and unpatchify
        patches = x[:, :N_img, :]
        out = patches.reshape(B, H_img, W_img, p, p, self.config.out_channels)
        out = out.transpose(0, 1, 3, 2, 4, 5)
        out = out.reshape(B, H_img * p, W_img * p, self.config.out_channels)

        return out
