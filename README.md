# ERNIE-Image MLX

Apple Silicon native inference engine for [ERNIE-Image](https://huggingface.co/PaddlePaddle/ERNIE-Image) (8B DiT), built on [MLX](https://github.com/ml-explore/mlx).

## Why

ERNIE-Image is an 8.03B parameter Diffusion Transformer that generates high-quality images from text prompts. The official pipeline runs on PyTorch with MPS backend, but only achieves ~25% GPU utilization on Apple Silicon due to Python-to-Metal kernel dispatch overhead.

This project rewrites the entire inference stack in MLX — Apple's native ML framework that schedules Metal kernels directly — to eliminate that overhead and unlock the full potential of Apple Silicon GPUs.

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

## Performance

Benchmarked on Mac M5 Pro (48GB), float32, CFG enabled (batch=2):

### Per-step latency

| Resolution | PyTorch MPS | MLX | Speedup |
|-----------|-------------|-----|---------|
| 768x432 | 4.2s | 3.56s | 1.18x |
| 1024x1024 | 19.0s | 12.1s | 1.57x |

### With CFG cutoff (cfg_cutoff=0.5)

CFG cutoff applies classifier-free guidance only during the first N% of denoising steps (structure formation), then switches to conditional-only prediction (B=1) for detail refinement — cutting compute nearly in half for late steps.

| Resolution | PyTorch MPS | MLX + cutoff | Speedup |
|-----------|-------------|--------------|---------|
| 1024x1024 | 19.0s/step | 10.6s/step | 1.79x |
| 1024x1024 (total 20 steps) | 380s | 213s | 1.78x |

### Without CFG (guidance_scale=1.0)

| Resolution | PyTorch MPS | MLX | Speedup |
|-----------|-------------|-----|---------|
| 768x432 | ~2.1s | 1.87s | 1.12x |
| 1024x1024 | ~9.5s | 6.1s | 1.56x |

## Optimizations Applied

| Optimization | Savings | Description |
|---|---|---|
| **QKV Fusion** | ~4ms/block (144ms/step) | Merge Q/K/V into single nn.Linear |
| **FFN Gate+Up Fusion** | ~9ms/block (330ms/step) | Merge gate_proj + up_proj into single nn.Linear |
| **No-mask SDPA** | ~6.5ms/block (234ms/step) | Skip attention mask when all text tokens are valid |
| **Cached prepare_inputs** | ~100ms/step | Precompute RoPE, text projection, attention mask once |
| **mx.compile** | ~8% | Operator fusion via MLX compilation |
| **CFG Cutoff** | up to 40% total time | Conditional-only prediction for late denoising steps |

### Optimizations tested and rejected

- **bfloat16**: Slower than float32 on Apple Silicon (lacks fast bf16 matmul)
- **float16**: NaN in AdaLN due to limited dynamic range
- **Manual matmul fusion**: `x @ weight.T` slower than `nn.Linear` (misses optimized Metal kernels)
- **INT4/INT8 quantization**: Works but excluded from default path by design choice

## Numerical Accuracy

Verified against PyTorch reference (float32, identical inputs):

| Component | Max Diff |
|---|---|
| Timestep embedding | 3e-5 |
| Patch embedding | 3e-5 |
| 3D RoPE | 1e-6 |
| Full 36-layer forward | 0.041 (0.34% relative) |
| Scheduler sigmas | 0.0 (exact) |

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

# Read prompt from file
python -m ernie_mlx.generate \
  --prompt-file prompt.txt \
  --height 768 --width 432

# Faster with CFG cutoff (first 50% steps use CFG, rest conditional-only)
python -m ernie_mlx.generate \
  --prompt "a cat sitting on a windowsill" \
  --steps 50 --cfg-cutoff 0.5

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

## Key Insights

1. **Compute-bound, not dispatch-bound**: Both PyTorch MPS and MLX achieve ~25% GPU ALU utilization. The bottleneck is raw TFLOPS throughput (21 TFLOPS needed per step, M5 Pro peak is 25 TFLOPS), not framework overhead.

2. **Fusion via nn.Linear matters**: Fusing QKV and FFN projections into single `nn.Linear` calls is significantly faster than manual `x @ weight.T`, because MLX's `nn.Linear` routes through optimized Metal kernels that raw matmul doesn't access.

3. **Attention mask elimination**: When all text tokens are valid (no padding), skipping the attention mask lets SDPA use a faster Metal kernel path — saving 6.5ms per layer.

4. **CFG is the cost multiplier**: Classifier-free guidance doubles the batch size and thus doubles all computation. CFG cutoff is the single biggest lever for wall-clock speedup without architectural changes.

## License

This project is for research and personal use. Model weights are subject to the [ERNIE-Image license](https://huggingface.co/PaddlePaddle/ERNIE-Image).
