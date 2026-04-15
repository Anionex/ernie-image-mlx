# MLX Optimization Research Log

## Baseline
- PyTorch MPS: 4.2s/step @ 768x432 with CFG (B=2)
- Target: <2.1s/step (2x speedup)

## Current MLX Performance
- Full forward (B=2, 768x432): **3364ms** (no-mask opt applied)
- Single DiT block: **97ms** (need ~49ms for 2x)
- Without CFG (B=1): **2.05s/step** — already near target!
- With CFG (B=2): **3.46s/step** — the problem is CFG doubles compute

## Per-Block Breakdown (97ms total)
| Component | Time | % |
|-----------|------|---|
| FFN (gate+up+down) | 53.6ms | 55% |
| Self-attention total | 41.5ms | 43% |
| - QKV projections | 19.8ms | 20% |
| - SDPA (no mask) | 8.1ms | 8% |
| - SDPA (with mask) | 14.6ms | 15% |
| - Output projection | 5.9ms | 6% |
| - RoPE + norms | ~8ms | 8% |
| Other (AdaLN etc) | ~2ms | 2% |

## Optimizations Applied
1. **No-mask SDPA**: When all text_lens == Tmax, skip attention mask. Saves ~6.5ms/layer.
   - Applied by setting text_lens = [Tmax, Tmax] in _pad_text (force_uniform_length=True)
2. **Cached prepare_inputs**: RoPE, mask, text projection computed once, reused across steps.
3. **mx.compile()**: Wraps transformer forward. Marginal or no improvement observed.

## Optimizations Tested But Rejected
1. **Fused gate+up FFN**: Manual `x @ weight.T` was SLOWER than two nn.Linear calls (112ms vs 100ms).
   nn.Linear has optimized Metal kernels that raw matmul doesn't use.
2. **bfloat16**: 4.18s/step — SLOWER than float32 (3.86s). Apple Silicon lacks fast bf16 matmul.
3. **float16**: NaN issues in AdaLN due to limited dynamic range.
4. **float16 weights with float32 input**: 5.9ms vs 5.5ms for single Linear — slightly SLOWER.
   Memory bandwidth is NOT the bottleneck.

## Promising Optimizations (Not Yet Applied)
1. **QKV fusion via nn.Linear**: Single [4096, 12288] vs 3x [4096, 4096].
   Benchmark: 15.8ms vs 19.8ms = saves **4ms/block = 144ms/step**.
   Need to implement as nn.Linear (not manual matmul) and remap weights post-load.

2. **True distillation-free CFG**: Run uncond+cond in a single batch-2 forward is what
   we already do. No further batching gain possible.

3. **Reduce float32 casts in AdaLN**: Currently 6 astype(float32) calls per block even
   when already float32. Benchmark showed negligible cost (1.08ms vs 1.00ms), skip.

4. **CFG-free generation**: guidance_scale=1.0 gives 2.05s/step. If image quality
   is acceptable, this alone achieves the 2x target.

5. **Fewer steps**: 20→15 steps could maintain quality with flow matching schedulers.

## Key Insight
The problem is fundamentally compute-bound, not dispatch-bound as originally hypothesized.
- 36 layers × (7 matmuls each) × [2, 1321, 4096/12288] = ~21 TFLOPS per step
- M5 Pro peak: 25 TFLOPS → theoretical minimum ~0.84s
- Current: 3.36s → **~25% compute efficiency** (same as PyTorch MPS!)
- MLX eliminates dispatch overhead but the GPU ALU utilization is the real limit

## TODO
- [ ] Implement QKV fusion (saves 144ms/step → 3.22s → still not 2x)
- [ ] Investigate if mx.compile actually helps (test with/without)
- [ ] Try INT4 + no-mask combo
- [ ] Profile GPU utilization with MLX metal profiling
- [ ] Consider architectural shortcuts: skip layers for early steps, distilled models
