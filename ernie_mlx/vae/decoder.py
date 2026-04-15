"""VAE Decoder for ERNIE-Image (MLX).

Standard Conv + ResNet + GroupNorm + Upsample decoder.
All convolutions use MLX NHWC format.
"""
import mlx.core as mx
import mlx.nn as nn


class GroupNorm(nn.Module):
    """Group Normalization (matches PyTorch nn.GroupNorm)."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, H, W, C] (NHWC)
        B, H, W, C = x.shape
        G = self.num_groups
        x = x.reshape(B, H, W, G, C // G)
        mean = mx.mean(x, axis=(1, 2, 4), keepdims=True)
        var = mx.var(x, axis=(1, 2, 4), keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        x = x.reshape(B, H, W, C)
        return x * self.weight + self.bias


class ResNetBlock(nn.Module):
    """ResNet block with GroupNorm and optional channel shortcut."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32):
        super().__init__()
        self.norm1 = GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_shortcut = None

    def __call__(self, x: mx.array) -> mx.array:
        residual = x

        x = self.norm1(x)
        x = nn.silu(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = nn.silu(x)
        x = self.conv2(x)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return x + residual


class AttentionBlock(nn.Module):
    """Self-attention block in VAE mid_block."""

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        self.group_norm = GroupNorm(num_groups, channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = [nn.Linear(channels, channels)]

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        residual = x

        x = self.group_norm(x)
        x = x.reshape(B, H * W, C)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Simple attention (no multi-head needed for VAE, channels are small)
        scale = C ** -0.5
        attn = (q @ k.transpose(0, 2, 1)) * scale
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v

        out = self.to_out[0](out)
        out = out.reshape(B, H, W, C)
        return out + residual


class Upsample(nn.Module):
    """2x nearest-neighbor upsample + conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        # Nearest neighbor 2x upsample
        x = mx.repeat(x, 2, axis=1)
        x = mx.repeat(x, 2, axis=2)
        x = self.conv(x)
        return x


class MidBlock(nn.Module):
    """Mid block: ResNet + Attention + ResNet."""

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        self.resnets = [
            ResNetBlock(channels, channels, num_groups),
            ResNetBlock(channels, channels, num_groups),
        ]
        self.attentions = [AttentionBlock(channels, num_groups)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class UpBlock(nn.Module):
    """Decoder up block: N resnets + optional upsample."""

    def __init__(self, in_channels: int, out_channels: int, num_resnets: int = 3,
                 has_upsample: bool = True, num_groups: int = 32):
        super().__init__()
        resnets = []
        for i in range(num_resnets):
            ch_in = in_channels if i == 0 else out_channels
            resnets.append(ResNetBlock(ch_in, out_channels, num_groups))
        self.resnets = resnets

        if has_upsample:
            self.upsamplers = [Upsample(out_channels)]
        else:
            self.upsamplers = None

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class VAEDecoder(nn.Module):
    """Full VAE decoder: post_quant_conv + conv_in + mid_block + up_blocks + conv_out.

    Includes batch norm denormalization (bn running_mean/var from training).
    """

    def __init__(self, latent_channels: int = 32,
                 block_out_channels: tuple = (128, 256, 512, 512),
                 layers_per_block: int = 2,
                 num_groups: int = 32):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)

        # Decoder mirrors encoder in reverse
        reversed_channels = list(reversed(block_out_channels))  # [512, 512, 256, 128]

        self.decoder = DecoderModule(
            latent_channels=latent_channels,
            block_out_channels=reversed_channels,
            layers_per_block=layers_per_block,
            num_groups=num_groups,
        )

    def __call__(self, latents: mx.array) -> mx.array:
        """
        Args:
            latents: [B, H, W, C] in NHWC, C=32 latent channels (already denormalized)
        Returns:
            [B, H*8, W*8, 3] decoded image in [0,1] range
        """
        x = self.post_quant_conv(latents)
        x = self.decoder(x)
        # Clamp to [0, 1]
        x = mx.clip(x / 2 + 0.5, 0.0, 1.0)
        return x


class DecoderModule(nn.Module):
    """The decoder sub-module (matching 'decoder.*' weight keys)."""

    def __init__(self, latent_channels: int, block_out_channels: list,
                 layers_per_block: int, num_groups: int):
        super().__init__()
        top_ch = block_out_channels[0]  # 512

        self.conv_in = nn.Conv2d(latent_channels, top_ch, kernel_size=3, padding=1)
        self.mid_block = MidBlock(top_ch, num_groups)

        # Up blocks: reverse channel progression
        up_blocks = []
        prev_ch = top_ch
        num_blocks = len(block_out_channels)
        for i, out_ch in enumerate(block_out_channels):
            has_upsample = (i < num_blocks - 1)
            up_blocks.append(UpBlock(
                in_channels=prev_ch,
                out_channels=out_ch,
                num_resnets=layers_per_block + 1,
                has_upsample=has_upsample,
                num_groups=num_groups,
            ))
            prev_ch = out_ch
        self.up_blocks = up_blocks

        self.conv_norm_out = GroupNorm(num_groups, block_out_channels[-1])
        self.conv_out = nn.Conv2d(block_out_channels[-1], 3, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        # Cast to float32 for VAE decode (force_upcast)
        x = x.astype(mx.float32)

        x = self.conv_in(x)
        x = self.mid_block(x)

        for block in self.up_blocks:
            x = block(x)

        x = self.conv_norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)
        return x
