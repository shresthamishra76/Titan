# Benchmark: matmul (Milestone 2 baseline)

Matrix multiply dominates transformer inference cost, so it is the first and
most important optimization target. This records the baseline (naive `ijk`) vs
the optimized kernel and the methodology behind the numbers.

## Setup

- **Machine:** Apple Silicon MacBook Air (arm64).
- **Compiler:** Apple clang 21, `-O3` (Release).
- **Harness:** `examples/benchmark.cpp` (`titan_benchmark [N]`), square `N×N×N`
  float32 matmul, one warmup + median of 5 runs. FLOPs counted as `2·N³`.

## Results

| N | naive `ijk` | titan matmul | speedup |
|---|-------------|--------------|---------|
| 512 | 2.87 GFLOP/s (93.6 ms) | **74.3 GFLOP/s** (3.6 ms) | **25.9×** |
| 1024 | 2.45 GFLOP/s (876 ms) | **90.0 GFLOP/s** (23.9 ms) | **36.7×** |

## What changed

Three compounding optimizations, all behind the same `titan::matmul` API and
verified identical (within f32 tolerance) to the naive reference by
`TensorTest.MatmulLargeMatchesNaiveReference`:

1. **Cache-friendly `ikj` ordering.** The naive `ijk` dot product strides down
   a column of `B` (cache-hostile). Reordering to `ikj` streams a row of `B` and
   the output row sequentially and accumulates — turning random access into
   linear access. This alone is the bulk of the win.
2. **NEON SIMD.** The inner `j` loop uses ARM NEON (`vfmaq_f32`) to fuse-multiply-add
   4 floats per instruction, with a scalar tail for the remainder.
3. **Thread pool.** Rows are split across a persistent `ThreadPool` (one worker
   per hardware thread) for matmuls above a size threshold, so small matmuls skip
   the threading overhead.

## Honest headroom

~90 GFLOP/s is a solid from-scratch result but still well below a tuned BLAS
(register/cache blocking reach several hundred GFLOP/s on this hardware). The
remaining Milestone 2 issues target that gap:

- **#18** register + L1/L2 cache blocking (tiling).
- **#17** profiling pass to confirm the next hotspot.
- **#21** scratch-arena allocation to cut per-op allocation in the forward pass.

End-to-end tokens/sec on a real model (issue #16 full) is captured once a
TinyLlama GGUF is available to run.
