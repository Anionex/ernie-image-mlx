"""Model configuration dataclasses."""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TransformerConfig:
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_layers: int = 36
    ffn_hidden_size: int = 12288
    in_channels: int = 128
    out_channels: int = 128
    patch_size: int = 1
    text_in_dim: int = 3072
    rope_theta: int = 256
    rope_axes_dim: Tuple[int, int, int] = (32, 48, 48)
    eps: float = 1e-6
    qk_layernorm: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass
class SchedulerConfig:
    num_train_timesteps: int = 1000
    shift: float = 3.0
    use_dynamic_shifting: bool = False
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096
    time_shift_type: str = "exponential"
    invert_sigmas: bool = False


@dataclass
class VAEConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 32
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    act_fn: str = "silu"
    mid_block_add_attention: bool = True
    patch_size: Tuple[int, int] = (2, 2)
    force_upcast: bool = True


@dataclass
class TextEncoderConfig:
    hidden_size: int = 3072
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    num_hidden_layers: int = 26
    intermediate_size: int = 9216
    vocab_size: int = 131072
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    head_dim: int = 128
    # YaRN RoPE parameters
    rope_scaling_factor: float = 16.0
    rope_original_max_pos: int = 16384
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
