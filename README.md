# ERNIE-Image MLX

**2x faster** ERNIE-Image 8B inference on Apple Silicon — native MLX rewrite of the full DiT pipeline, with zero accuracy loss.

<p align="center">
  <img src="assets/sample_1024x1024.png" width="512" alt="Sample output: 1024x1024, 20 steps, cfg_cutoff=0.5, seed=42"/>
  <br>
  <em>1024x1024, 20 steps — generated in ~192s (MLX) vs ~380s (PyTorch MPS)</em>
</p>

## Results

Benchmarked on Mac M5 Pro (48GB), float32, no quantization:

| | PyTorch MPS | MLX | Improvement |
|---|---|---|---|
| **Speed** (1024x1024, 20 steps) | 19.0s/step | **9.5s/step** | **2.0x faster** |
| **Memory** (denoising phase) | ~50GB peak | **27GB** | **46% less** |
| **Accuracy** (vs PyTorch f32) | baseline | max diff 0.041 | **<0.35% relative** |

> MLX dynamically frees the text encoder (~7GB) after encoding, keeping denoising memory well within 48GB. PyTorch MPS holds all models simultaneously.

### Detailed benchmarks

**With CFG cutoff (recommended)**

| Resolution | PyTorch MPS | MLX (cfg_cutoff=0.5) | Speedup |
|---|---|---|---|
| 1024x1024 | 19.0s/step | 9.5s/step | **2.0x** |

**Full CFG (all steps B=2)**

| Resolution | PyTorch MPS | MLX | Speedup |
|---|---|---|---|
| 768x432 | 4.2s/step | 3.56s/step | 1.18x |
| 1024x1024 | 19.0s/step | 11.0s/step | 1.73x |

**No CFG (guidance_scale=1.0)**

| Resolution | PyTorch MPS | MLX | Speedup |
|---|---|---|---|
| 768x432 | ~2.1s/step | 1.87s/step | 1.12x |
| 1024x1024 | ~9.5s/step | 5.8s/step | 1.64x |

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.12+
- [ERNIE-Image model weights](https://huggingface.co/PaddlePaddle/ERNIE-Image) downloaded to `models/PaddlePaddle/ERNIE-Image/`

### Install

```bash
pip install -e ".[mlx]"
```

### Generate

```bash
# Basic generation
python -m ernie_mlx.generate \
  --prompt "a serene mountain landscape at sunset" \
  --height 1024 --width 1024 \
  --steps 50 --guidance-scale 4.0 \
  --seed 42 --output output.png

# Faster with CFG cutoff (first 50% steps use CFG, rest conditional-only)
python -m ernie_mlx.generate \
  --prompt "a cat sitting on a windowsill" \
  --steps 50 --cfg-cutoff 0.5

# Read prompt from file
python -m ernie_mlx.generate \
  --prompt-file prompt.txt \
  --height 768 --width 432

# No CFG (fastest, but lower prompt adherence)
python -m ernie_mlx.generate \
  --prompt "a landscape" \
  --guidance-scale 1.0 --steps 20
```

### Python API

```python
from ernie_mlx import ErnieImagePipeline

pipe = ErnieImagePipeline("models/PaddlePaddle/ERNIE-Image")
pipe.load()

image = pipe(
    prompt="a beautiful sunset over the ocean",
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=4.0,
    cfg_cutoff=0.5,
    seed=42,
)
image.save("output.png")
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `height` / `width` | 768 / 432 | Output resolution (must be divisible by 16) |
| `num_inference_steps` | 20 | Denoising steps (50 recommended for best quality) |
| `guidance_scale` | 5.0 | CFG scale (4.0 matches official default) |
| `cfg_cutoff` | 1.0 | Fraction of steps using CFG (0.5 = first 50%) |
| `seed` | None | Random seed for reproducibility |

## Architecture

```
Text Prompt
    |
    v
[Mistral-3 Text Encoder]  3.4B params, 26-layer GQA Transformer
    |                      YaRN RoPE, hidden=3072, 32 heads / 8 KV heads
    v
[DiT Transformer]          8.0B params, 36-layer DiT
    |                      3D RoPE, Shared AdaLN, hidden=4096, 32 heads
    |                      Gated FFN (gate + up + down), QK-norm
    v
[VAE Decoder]              Conv + ResNet + GroupNorm + Upsample
    |                      4 UpBlocks, BN denormalization
    v
  Image
```

## Optimizations

### Applied

| Optimization | Impact | Description |
|---|---|---|
| **Single compiled function** | eliminates 30s switch cost | Only compile CFG (B=2) path; cond (B=1) runs uncompiled at same speed. Avoids GPU cache thrashing from two compiled fns on unified memory. |
| **Precomputed RoPE cos/sin** | saves 72 trig ops/step | Compute cos/sin once in `prepare_inputs`, reuse across all 36 layers. |
| **QKV Fusion** | ~4ms/block (144ms/step) | Merge Q/K/V into single `nn.Linear` |
| **FFN Gate+Up Fusion** | ~9ms/block (330ms/step) | Merge gate_proj + up_proj into single `nn.Linear` |
| **No-mask SDPA** | ~6.5ms/block (234ms/step) | Skip attention mask when all text tokens are valid |
| **Cached prepare_inputs** | ~100ms/step | Precompute RoPE, text projection, attention mask once |
| **mx.compile (CFG only)** | ~8% for B=2 path | Operator fusion via MLX compilation |
| **CFG Cutoff** | up to 40% total time | Conditional-only prediction for late denoising steps |
| **Text encoder freeing** | recovers ~7GB | Free text encoder after encoding to avoid memory pressure during denoising |

### Tested and rejected

| Optimization | Result | Reason |
|---|---|---|
| **Dual compiled functions** | 30-47s switch cost per transition | Two `mx.compile`'d fns thrash GPU cache on unified memory (64GB active on 48GB system) |
| **bfloat16** | slower | Apple Silicon lacks fast bf16 matmul |
| **float16** | NaN | Limited dynamic range causes NaN in AdaLN |
| **Manual `x @ weight.T`** | slower | Misses optimized Metal kernels that `nn.Linear` uses |
| **mx.clear_cache()** | destroys compiled traces | Forces 30-40s retrace on next compiled call |
| **Pre-eval concatenate** | slower | Breaks lazy evaluation fusion between concat and model forward |
| **Cache limit (mx.set_cache_limit)** | slower | Restricts MLX memory pool, hurts allocation performance |

## Numerical Accuracy

Verified against PyTorch reference (float32, identical inputs):

| Component | Max Diff |
|---|---|
| Timestep embedding | 3e-5 |
| Patch embedding | 3e-5 |
| 3D RoPE | 1e-6 |
| Full 36-layer forward | 0.041 (0.34% relative) |
| Scheduler sigmas | 0.0 (exact) |

## Key Insights

1. **GPU cache thrashing is the #1 bottleneck on unified memory**: Having two `mx.compile`'d functions whose caches exceed physical RAM causes 30-47s switching costs — worse than no compilation at all. The fix: compile only the expensive path (B=2), run the cheap path (B=1) uncompiled.

2. **Fusion via nn.Linear matters**: Fusing QKV and FFN projections into single `nn.Linear` calls is significantly faster than manual `x @ weight.T`, because MLX's `nn.Linear` routes through optimized Metal kernels.

3. **Attention mask elimination**: When all text tokens are valid (no padding), skipping the attention mask lets SDPA use a faster Metal kernel path — saving 6.5ms per layer.

4. **CFG is the cost multiplier**: Classifier-free guidance doubles the batch size and thus doubles all computation. CFG cutoff is the single biggest lever for wall-clock speedup without architectural changes.

5. **Never call `mx.clear_cache()` with compiled functions**: It invalidates compiled GPU kernel caches, forcing 30-40s retrace on the next call.

## Project Structure

```
ernie_mlx/
  __init__.py              # Package entry point
  pipeline.py              # Full inference pipeline
  config.py                # Model configuration dataclasses
  scheduler.py             # FlowMatch Euler scheduler
  tokenizer.py             # HuggingFace tokenizers wrapper
  generate.py              # CLI entry point
  model/
    transformer.py          # 36-layer DiT with Shared AdaLN
    dit_block.py            # Single transformer block
    attention.py            # Self-Attention + QKV fusion + RoPE + QK-norm
    feed_forward.py         # Gated FFN + gate/up fusion
    embeddings.py           # 3D RoPE, TimestepEmbedding, PatchEmbed
  text_encoder/
    mistral.py              # Mistral-3 text encoder (3.4B)
  vae/
    decoder.py              # VAE decoder (Flux2-style)
  weights/
    loader.py               # safetensors weight loading
    key_maps.py             # PyTorch -> MLX weight key remapping
```

## License

This project is for research and personal use. Model weights are subject to the [ERNIE-Image license](https://huggingface.co/PaddlePaddle/ERNIE-Image).
