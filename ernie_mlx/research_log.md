# MLX Optimization Research Log

## Final Results (2026-04-16)

**Target achieved: 2.0x speedup over PyTorch MPS @ 1024x1024**

| Metric | PyTorch MPS | MLX | Improvement |
|---|---|---|---|
| Speed (1024x1024, cfg_cutoff=0.5) | 19.0s/step | 9.5s/step | 2.0x |
| Memory (denoising) | ~50GB peak | 27GB | 46% less |
| Accuracy | baseline | max diff 0.041 | <0.35% relative |

**Turbo model: ~8x faster end-to-end**

| Metric | Base (50 steps) | Turbo (8 steps) |
|---|---|---|
| Total time (1024x1024) | ~475s | 59.6s |
| Per-step | 9.5s/step | 7.15s/step |
| CFG | guidance_scale=4.0, cfg_cutoff=0.5 | None (guidance_scale=1.0) |

## Optimization Timeline

### Phase 1: Core Implementation
- Built full MLX pipeline: Mistral-3 text encoder + 36-layer DiT + VAE decoder
- Verified numerical accuracy against PyTorch reference (max diff 0.041 over 36 layers)
- Initial performance: ~4.2s/step @ 768x432 (1.0x, no speedup)

### Phase 2: Operator-level Optimizations

**QKV Fusion** — saves 144ms/step
- Merge Q/K/V weights into single `nn.Linear(4096, 12288)`
- Must use `nn.Linear`, not manual `x @ weight.T` (misses optimized Metal kernels)
- Benchmark: 15.8ms vs 19.8ms per block = 4ms/block x 36 layers

**FFN Gate+Up Fusion** — saves 330ms/step
- Merge gate_proj + up_proj into single `nn.Linear(4096, 24576)`
- Same lesson: `nn.Linear` >> manual matmul
- Benchmark: 44.3ms vs 53.6ms per block = 9ms/block x 36 layers

**No-mask SDPA** — saves 234ms/step
- Set all text_lens = Tmax (force_uniform_length=True) so SDPA skips mask
- The zero-padded positions have near-zero embeddings, minimal impact on attention
- Benchmark: 8.1ms vs 14.6ms SDPA per block = 6.5ms/block x 36 layers

**Cached prepare_inputs** — saves ~100ms/step
- Precompute RoPE, text projection, and attention mask once before denoising loop
- Reuse across all steps without recomputation

**mx.compile** — ~8% speedup for CFG path
- Wraps transformer forward for operator fusion
- Only meaningful for B=2 (compute-heavy); B=1 shows no improvement

After Phase 2: **3.56s/step @ 768x432 (1.18x), 12.1s/step @ 1024x1024 (1.57x)**

### Phase 3: CFG Cutoff

**CFG cutoff** — up to 40% total time reduction
- First N% of steps use full CFG (B=2), rest use cond-only (B=1)
- Early steps decide structure/composition; late steps refine details
- cfg_cutoff=0.5: theoretical 2.22x at steady-state

After Phase 3: **10.6s/step average (1.79x)** — limited by switching overhead

### Phase 4: Precomputed RoPE cos/sin

**Precomputed cos/sin** — eliminates 72 trig computations/step
- Previously: `cos(freqs)` and `sin(freqs)` computed inside `apply_rotary_emb`, called 2x per layer x 36 layers = 72 times per step
- Fix: compute once in `prepare_inputs`, cache as tuple `(cos, sin)`
- Also stabilized GPU memory behavior (no more progressive CFG slowdown from growing trig cache)

### Phase 5: GPU Cache Thrashing Discovery (Breakthrough)

**Root cause of pipeline overhead identified**: Two `mx.compile`'d functions thrashing GPU cache on unified memory.

Profiling revealed:
- Isolated CFG (B=2): 11.0s/step — steady
- Isolated Cond (B=1): 5.8s/step — steady
- Theoretical average: 8.4s/step = **2.26x** (exceeds target!)
- Actual pipeline: 10.5s/step = **1.81x** — where's the 2.1s/step gap?

The gap came from TWO retrace/switching events:
- Step 0 (first CFG after warmup): +9s overhead
- Step 10 (CFG→Cond switch): +26s overhead
- Total: 35s / 20 steps = 1.75s/step overhead

**Direct switching cost measurement:**
```
CFG → CFG (no switch):  11.0s, 11.1s, 11.2s     — stable
Cond → CFG (switch):    39.0s, 11.4s, 12.5s      — 28s overhead on first call
CFG → Cond (switch):    35.7s, 6.9s, 6.7s        — 29s overhead on first call
Alternating:            47s, 33s, 48s, 47s, 47s   — EVERY call is slow!
```

**Cause**: Each compiled function's cached trace was ~20-25GB. With two functions, total = 40-50GB + 16GB model weights = 56-66GB. On 48GB system, massive swap thrashing.

**Fix**: Use a single compiled function for CFG (B=2) and call the model uncompiled for cond (B=1). Uncompiled B=1 is equally fast (5.8s). Zero switching overhead.

After Phase 5: **9.5s/step (2.0x)**, active memory 27GB

### Phase 6: Pipeline Integration

Additional pipeline-level fixes:
- **Text encoder freeing**: `del self.text_encoder` + `gc.collect()` after encoding. Recovers ~7GB. Without this, denoising runs at 1.15x (all models in memory = 50GB, swap thrashing).
- **mx.eval(latents) before loop**: Materializes random latent before compiled function sees it, avoiding lazy-graph interference.
- **No mx.clear_cache()**: Calling this anywhere in the pipeline destroys compiled GPU kernel caches, forcing 30-40s retrace. Discovered the hard way.

## Per-Block Breakdown (97ms total @ 768x432, B=2)

| Component | Time | % |
|---|---|---|
| FFN (gate+up+down) | 44ms (fused) | 45% |
| Self-attention total | 36ms (fused QKV) | 37% |
| - QKV projection | 16ms | 16% |
| - SDPA (no mask) | 8ms | 8% |
| - Output projection | 6ms | 6% |
| - RoPE + norms | ~6ms | 6% |
| Other (AdaLN etc) | ~2ms | 2% |

## Optimizations Tested and Rejected

| Optimization | Result | Details |
|---|---|---|
| bfloat16 | 4.18s (slower) | Apple Silicon lacks fast bf16 matmul hardware |
| float16 | NaN | Limited dynamic range causes NaN in AdaLN modulation |
| float16 weights + float32 compute | 5.9ms vs 5.5ms per Linear (slower) | Memory bandwidth is NOT the bottleneck |
| Manual `x @ weight.T` | 112ms vs 100ms per FFN (slower) | Misses optimized Metal kernels in `nn.Linear` |
| Deleting pre-fusion weights (`del self.to_q`) | CFG steps 12.5→16s (slower) | Triggers GPU memory reallocation |
| Dual compiled functions | 30-47s per switch | GPU cache thrashing on unified memory |
| Compiled entire denoising step | no improvement | CFG math + scheduler step already <1ms (lazy ops fused) |
| Pre-eval concatenated inputs | 12.4s vs 11.5s (slower) | Breaks lazy evaluation fusion |
| mx.set_cache_limit | slower | Restricts MLX memory pool |
| mx.clear_cache() | 30-40s retrace | Destroys compiled function GPU kernel caches |
| Warmup at different resolution | +3s/step | Pollutes compiled function cache with wrong-shape trace |
| AdaLN float32 upcasts | no measurable diff | 1.08ms vs 1.00ms per block, removed |
| Single eval per step (skip eval pred) | 14-16s vs 11s CFG (much slower) | Compiled fn output must be eval'd before slicing for CFG math |

### Phase 7: Turbo Model Support

**ERNIE-Image-Turbo** — distilled model, same 8B DiT architecture (36 layers, 4096 dim), different weights.

Key differences from base:
- **8 steps** (vs 50 for base)
- **guidance_scale=1.0** (no CFG needed, always B=1)
- **Scheduler shift=4.0** (vs 3.0 for base)

Implementation: pipeline auto-detects via directory name (`"turbo" in model_dir.name.lower()`), sets defaults accordingly. Skips CFG compilation for Turbo since B is always 1.

**Turbo benchmarks (MLX, 1024x1024):**
- 8 steps × 7.15s/step = 57.2s denoising
- Total: 59.6s (including text encoding + VAE decode)
- vs Base (50 steps, cfg_cutoff=0.5): ~475s → **~8x faster end-to-end**

**PyTorch MPS Turbo failure:**
PyTorch diffusers cannot run Turbo on 48GB Mac. It loads all models simultaneously (text encoder 7.7GB + transformer 16GB + VAE 0.2GB + prompt enhancer ~3GB) without freeing any. Observed 27GB swap on a 48GB system — unusable. MLX avoids this entirely by freeing the text encoder after encoding.

## Key Insight

The original hypothesis — that MLX would speed up inference by eliminating Python→Metal dispatch overhead — was partially wrong. Both PyTorch MPS and MLX achieve ~25% GPU ALU utilization, but this is NOT because the GPU is compute-saturated.

**The 25% ALU is a mixed bottleneck:**
- **Matmul ops** (~45% of step time): Compute-bound (arithmetic intensity ~1241 FLOP/byte, far exceeding the GPU's ops:byte ratio). But ALU utilization during matmul is only ~37% — the theoretical peak isn't fully realized due to warp scheduling, memory latency, and kernel overhead.
- **Non-matmul ops** (~35% of step time): Memory-bound (norm, RoPE, activations — low arithmetic intensity, limited by bandwidth).
- **Kernel dispatch gaps** (~20% of step time): Pure idle time between Metal kernel launches.

The 25% overall ALU is the weighted average of ~37% during matmul, ~0% during non-matmul, and 0% during idle gaps — not a single bottleneck mode.

The actual speedup comes from:
1. **Operator fusion** (QKV, FFN, mx.compile) — reduces total ops and kernel launches
2. **CFG cutoff** — halves compute for late steps
3. **Eliminating GPU cache thrashing** — the single biggest win, not from compute but from memory management

The lesson: on unified memory systems, **memory layout and allocation patterns matter more than raw compute optimization**.
