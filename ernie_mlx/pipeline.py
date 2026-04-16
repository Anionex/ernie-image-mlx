"""ERNIE-Image MLX inference pipeline."""
import gc
import time
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from .config import TransformerConfig, TextEncoderConfig, VAEConfig, SchedulerConfig
from .model import ErnieImageTransformer
from .text_encoder import MistralTextEncoder
from .vae import VAEDecoder
from .scheduler import FlowMatchEulerScheduler
from .tokenizer import ErnieTokenizer
from .weights.loader import load_transformer_weights, load_text_encoder_weights, load_vae_weights


class ErnieImagePipeline:
    """Full ERNIE-Image inference pipeline in MLX.

    Flow:
        1. Tokenize + encode text prompt (Mistral3)
        2. Initialize random latent noise [B, 128, H/16, W/16]
        3. Denoise loop (FlowMatch Euler)
        4. BN denormalize + unpatchify
        5. VAE decode → pixel image
    """

    def __init__(self, model_dir: str):
        """
        Args:
            model_dir: path to model directory (base or turbo)
        """
        self.model_dir = Path(model_dir)
        self.vae_scale_factor = 16  # 2^4 (4 up blocks in VAE)
        self._loaded = False

        # Auto-detect turbo variant from directory name or config
        self.is_turbo = "turbo" in self.model_dir.name.lower()

        # Load BN stats immediately (tiny)
        vae_weights = mx.load(str(self.model_dir / "vae" / "diffusion_pytorch_model.safetensors"))
        self.bn_mean = vae_weights["bn.running_mean"]  # [128]
        self.bn_var = vae_weights["bn.running_var"]      # [128]

    @staticmethod
    def _cast_model(model: nn.Module, dtype):
        """Cast all model parameters to a given dtype."""
        params = nn.utils.tree_flatten(model.parameters())
        cast_params = [(k, v.astype(dtype)) for k, v in params]
        model.load_weights(cast_params)

    def load(self, dtype=mx.float32, quantize_bits: int = 0):
        """Load all model weights.

        Args:
            dtype: computation dtype for text encoder and transformer.
                   VAE always uses float32.
            quantize_bits: 0 for no quantization, 4 or 8 for INT4/INT8.
                          Reduces transformer memory from 15GB to ~4GB (INT4).
        """
        if self._loaded:
            return

        self._dtype = dtype
        t0 = time.time()
        print("Loading tokenizer...")
        tokenizer_path = self.model_dir / "tokenizer" / "tokenizer.json"
        self.tokenizer = ErnieTokenizer(str(tokenizer_path))

        print("Loading text encoder (3.4B)...")
        self.text_encoder = MistralTextEncoder()
        load_text_encoder_weights(self.text_encoder, str(self.model_dir / "text_encoder"))

        print("Loading transformer (8.0B)...")
        self.transformer = ErnieImageTransformer()
        load_transformer_weights(self.transformer, str(self.model_dir / "transformer"))

        # Fuse projections for faster matmul
        self.transformer.fuse_qkv_weights()
        self.transformer.fuse_ffn_weights()

        if quantize_bits > 0:
            print(f"Quantizing transformer to INT{quantize_bits}...")
            nn.quantize(self.transformer, bits=quantize_bits, group_size=64)

        print("Loading VAE decoder...")
        self.vae = VAEDecoder()
        load_vae_weights(self.vae, str(self.model_dir / "vae"))

        scheduler_config = SchedulerConfig(shift=4.0 if self.is_turbo else 3.0)
        self.scheduler = FlowMatchEulerScheduler(scheduler_config)

        if self.is_turbo:
            # Turbo uses guidance_scale=1.0 (no CFG), so no compiled function needed.
            self._compiled_cfg = None
        else:
            # Only compile CFG (B=2) path — it's compute-heavy and benefits from
            # kernel fusion. Cond (B=1) runs uncompiled at the same speed, and
            # having a single compiled fn avoids catastrophic GPU cache thrashing
            # that two compiled fns cause on unified memory (~30s switching cost).
            def _fwd_cfg(h, t, txt, tl, c):
                return self.transformer(h, t, txt, tl, c)
            self._compiled_cfg = mx.compile(_fwd_cfg)

        self._loaded = True
        t1 = time.time()
        variant = "turbo" if self.is_turbo else "base"
        print(f"All models loaded in {t1-t0:.1f}s (dtype={dtype}, variant={variant})")

    def _ensure_text_encoder(self):
        """Reload text encoder if it was freed after a previous generation."""
        if self.text_encoder is None:
            self.text_encoder = MistralTextEncoder()
            load_text_encoder_weights(self.text_encoder, str(self.model_dir / "text_encoder"))

    def encode_prompt(self, prompt: str) -> mx.array:
        """Encode a text prompt to hidden states.

        Returns:
            [T, hidden_size] text embeddings from second-to-last layer
        """
        self._ensure_text_encoder()
        input_ids = self.tokenizer.encode(prompt)
        if len(input_ids) == 0:
            input_ids = [1]  # BOS fallback
        input_ids = mx.array([input_ids])
        hidden = self.text_encoder(input_ids)  # [1, T, 3072]
        mx.eval(hidden)
        return hidden[0]  # [T, 3072]

    def _pad_text(self, text_hiddens: List[mx.array], text_in_dim: int = 3072,
                  force_uniform_length: bool = True):
        """Pad variable-length text hidden states to a batch.

        Args:
            text_hiddens: list of [T_i, dim] tensors
            force_uniform_length: if True, set all text_lens to Tmax so the
                attention mask is all-valid. This enables the fast no-mask SDPA
                path (~6.5ms/layer faster). The zero-padded positions have near-zero
                embeddings and minimal impact on attention.
        Returns:
            text_bth: [B, Tmax, dim]
            text_lens: [B]
        """
        B = len(text_hiddens)
        lens = [t.shape[0] for t in text_hiddens]
        Tmax = max(lens)
        text_bth = mx.zeros((B, Tmax, text_in_dim))
        for i, t in enumerate(text_hiddens):
            text_bth = text_bth.at[i, :t.shape[0], :].add(t)

        if force_uniform_length:
            # All elements get Tmax so SDPA skips the mask entirely
            text_lens = mx.array([Tmax] * B)
        else:
            text_lens = mx.array(lens)
        return text_bth, text_lens

    @staticmethod
    def _unpatchify(latents: mx.array) -> mx.array:
        """Reverse patchify: [B, H/2, W/2, 128] → [B, H, W, 32] (NHWC).

        Undoes the 2x2 spatial packing that maps 32 channels to 128.
        """
        B, Hp, Wp, C = latents.shape
        # [B, Hp, Wp, 32, 2, 2]
        x = latents.reshape(B, Hp, Wp, C // 4, 2, 2)
        # [B, Hp, 2, Wp, 2, 32]
        x = x.transpose(0, 1, 4, 2, 5, 3)
        # [B, H, W, 32]
        x = x.reshape(B, Hp * 2, Wp * 2, C // 4)
        return x

    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 768,
        width: int = 432,
        num_inference_steps: int = None,
        guidance_scale: float = None,
        cfg_cutoff: float = 1.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate an image from a text prompt.

        Args:
            prompt: text description
            negative_prompt: negative text description
            height: output image height (must be divisible by 16)
            width: output image width (must be divisible by 16)
            num_inference_steps: number of denoising steps (default: 8 for turbo, 20 for base)
            guidance_scale: CFG scale (default: 1.0 for turbo, 5.0 for base)
            cfg_cutoff: fraction of steps to use CFG (0.0-1.0). E.g. 0.5 means
                       first 50% of steps use full CFG (B=2), rest use B=1.
                       Early steps decide structure; late steps refine details.
            seed: random seed
        Returns:
            PIL Image
        """
        self.load()

        # Apply model-appropriate defaults
        if num_inference_steps is None:
            num_inference_steps = 8 if self.is_turbo else 20
        if guidance_scale is None:
            guidance_scale = 1.0 if self.is_turbo else 5.0

        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(f"Height and width must be divisible by {self.vae_scale_factor}")

        do_cfg = guidance_scale > 1.0
        cfg_steps = int(num_inference_steps * cfg_cutoff) if do_cfg else 0

        # 1. Encode text
        t_start = time.time()
        print(f"Encoding prompt ({len(prompt)} chars)...")
        cond_hidden = self.encode_prompt(prompt)
        if do_cfg:
            uncond_hidden = self.encode_prompt(negative_prompt)
            text_hiddens_cfg = [uncond_hidden, cond_hidden]
        else:
            text_hiddens_cfg = [cond_hidden]

        text_bth_cfg, text_lens_cfg = self._pad_text(text_hiddens_cfg)
        text_bth_cfg = text_bth_cfg.astype(self._dtype)

        # Prepare B=1 (cond-only) inputs for post-cutoff steps
        text_hiddens_cond = [cond_hidden]
        text_bth_cond, text_lens_cond = self._pad_text(text_hiddens_cond)
        text_bth_cond = text_bth_cond.astype(self._dtype)

        # Free text encoder (~7GB) — no longer needed after encoding.
        # Note: do NOT call mx.clear_cache() here — it would invalidate
        # compiled function GPU kernel caches and force expensive retraces.
        del self.text_encoder
        self.text_encoder = None
        gc.collect()

        t_encode = time.time()
        print(f"Text encoding: {t_encode - t_start:.2f}s")

        # 2. Initialize latents (patchified space)
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_channels = 128

        if seed is not None:
            mx.random.seed(seed)

        B = 1
        latents = mx.random.normal((B, latent_h, latent_w, latent_channels)).astype(self._dtype)
        mx.eval(latents)  # materialize before compiled function sees it

        # 3. Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # 4. Precompute cached inputs
        H_img = latent_h
        W_img = latent_w

        # Precompute cached inputs for both paths
        if cfg_steps > 0:
            cached_cfg = self.transformer.prepare_inputs(2, H_img, W_img, text_bth_cfg, text_lens_cfg)
            cos_cfg, sin_cfg = cached_cfg['rotary_pos_emb']
            mx.eval(cos_cfg, sin_cfg, cached_cfg['text_projected'])
            if cached_cfg['attention_mask'] is not None:
                mx.eval(cached_cfg['attention_mask'])

        cached_cond = self.transformer.prepare_inputs(1, H_img, W_img, text_bth_cond, text_lens_cond)
        cos_cond, sin_cond = cached_cond['rotary_pos_emb']
        mx.eval(cos_cond, sin_cond, cached_cond['text_projected'])
        if cached_cond['attention_mask'] is not None:
            mx.eval(cached_cond['attention_mask'])

        # 5. Denoising loop
        cfg_mode = f"cutoff={cfg_cutoff} ({cfg_steps}/{num_inference_steps} steps)" if cfg_steps < num_inference_steps else "full"
        print(f"Denoising {num_inference_steps} steps @ {width}x{height} (CFG {cfg_mode})...")

        cfg_freed = False
        for i in range(num_inference_steps):
            t = self.scheduler.timesteps[i:i+1]
            step_start = time.time()

            if i < cfg_steps:
                # Early steps: full CFG (B=2) — decides structure & composition
                latent_input = mx.concatenate([latents, latents], axis=0)
                t_batch = mx.broadcast_to(t, (2,))
                pred = self._compiled_cfg(latent_input, t_batch, text_bth_cfg, text_lens_cfg, cached_cfg)
                pred_uncond = pred[0:1]
                pred_cond = pred[1:2]
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                step_label = "CFG"
            else:
                # Free CFG inputs on first cond-only step
                if not cfg_freed and cfg_steps > 0:
                    del cached_cfg, text_bth_cfg, text_lens_cfg
                    cfg_freed = True
                # Late steps: cond-only (B=1) — use uncompiled model directly.
                # B=1 is equally fast uncompiled, and avoids GPU cache thrashing
                # from switching between two compiled functions.
                pred = self.transformer(latents, t, text_bth_cond, text_lens_cond, cached_cond)
                step_label = "cond"

            # Scheduler step
            latents = self.scheduler.step(pred, i, latents)
            mx.eval(latents)

            step_time = time.time() - step_start
            if i == 0 or (i + 1) % 5 == 0 or i == num_inference_steps - 1:
                label = f" [{step_label}]" if step_label else ""
                print(f"  Step {i+1}/{num_inference_steps}: {step_time:.2f}s{label}")

        t_denoise = time.time()
        total_denoise = t_denoise - t_encode
        print(f"Denoising complete: {total_denoise:.1f}s ({total_denoise/num_inference_steps:.2f}s/step)")

        # 7. BN denormalize (in patchified space, 128 channels)
        bn_std = mx.sqrt(self.bn_var + 1e-5)
        latents = latents * bn_std + self.bn_mean

        # 8. Unpatchify: [B, H/16, W/16, 128] → [B, H/8, W/8, 32]
        latents = self._unpatchify(latents)

        # 9. VAE decode
        print("VAE decoding...")
        t_vae_start = time.time()
        images = self.vae(latents)
        mx.eval(images)
        t_vae = time.time() - t_vae_start
        print(f"VAE decode: {t_vae:.2f}s")

        # 10. Convert to PIL
        img_np = np.array(images[0])  # [H, W, 3] in [0, 1]
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

        total_time = time.time() - t_start
        print(f"Total generation time: {total_time:.1f}s")

        return Image.fromarray(img_np)
