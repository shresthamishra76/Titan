# Titan

A production-style LLM inference runtime written from scratch in modern C++ — no prior C++ experience going in. The goal is to understand every layer between an HTTP request and generated tokens: tensor math, transformer execution, and eventually a scheduler with an OpenAI-compatible API, the same shape of system that sits behind vLLM, TGI, or an internal serving stack at a frontier lab.

- **[PRD.md](PRD.md)** — V1 scope (CPU inference), target end of August 2026
- **[PRD-v2.md](PRD-v2.md)** — V2 scope (scheduler, continuous batching, OpenAI-compatible API, dashboard)
- **[docs/architecture](docs/architecture)** — per-subsystem design docs: overview, rationale, diagrams, interfaces, tradeoffs

## Status

**Milestone 1 (CPU Runtime) — complete.** Titan loads a Llama-family GGUF model and generates text end-to-end on CPU: a tensor engine, a SentencePiece BPE tokenizer, an mmap-based GGUF loader, a transformer runtime (RoPE, grouped-query attention, SwiGLU), and a sampling engine, wired together by a generation CLI. 51 unit tests pass in CI on every push.

**Milestone 2 (Optimization) — next.** Benchmark harness, profiling, cache-blocked matmul, ARM NEON SIMD, memory pooling. See issues #16–#21.

**V2 (serving layer).** KV cache, request scheduler, continuous batching, OpenAI-compatible API, dashboard. See [PRD-v2.md](PRD-v2.md) and issues #22–#23.

## Engineering Highlights

- **Hyperparameters are read, never hard-coded.** Layer count, head counts, hidden/FFN dims, RoPE base, RMSNorm epsilon, context length, and vocab size are all pulled from GGUF metadata at load time (`runtime/models/GGUF.cpp`), so the runtime isn't wired to one specific model shape.
- **Zero-copy model loading.** `MappedFile` (`runtime/models/GGUF.h`) memory-maps the GGUF file read-only via `mmap`/`munmap` instead of reading weights into a heap buffer — tensor views point directly into the mapped region.
- **Grouped-query attention implemented explicitly, not borrowed from a library.** `runtime/transformer/Model.cpp` derives the query/KV head grouping (`group = n_head / n_head_kv`) and maps each query head to its shared KV head by hand, alongside RoPE and a pre-norm SwiGLU feed-forward block.
- **Matmul is parallelized behind a thread pool, gated by problem size.** `Tensor::matmul` only dispatches to `ThreadPool::parallel_for` (`runtime/tensor/ThreadPool.cpp`) when the work is large enough to amortize threading overhead — otherwise it runs single-threaded, avoiding pool overhead on small ops.
- **Every subsystem has a paired architecture doc** — overview, design rationale, interfaces, tradeoffs, and references — written alongside the implementation rather than after the fact (see [docs/architecture](docs/architecture)).

## Architecture (V1)

```
        Generation Loop (examples/generate)
                   │
        Transformer Runtime         runtime/transformer
                   │
   RoPE · GQA Attention · SwiGLU FFN · RMSNorm
                   │
        Tensor Engine               runtime/tensor
                   │
             CPU (arm64)
```

The tokenizer feeds token IDs into the runtime; the GGUF loader supplies the mmap'd weights. Target model is TinyLlama (Llama-family, GGUF, F16/F32) — because the target architecture is Llama, the runtime implements the Llama operator set specifically: RMSNorm (not LayerNorm), SiLU/SwiGLU (not GELU), GQA, and RoPE.

Everything above the tensor engine in the V2 diagram — KV cache, scheduler, continuous batching, OpenAI-compatible API, dashboard — is planned, not built. See [PRD-v2.md](PRD-v2.md) for the full V2 design.

## Tech Stack

C++20, CMake, GoogleTest, GitHub Actions CI. No ML framework, no BLAS library, no external tensor library — the tensor engine, GGUF parser, tokenizer, and transformer math are all implemented directly.

## Build

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

CI (`.github/workflows/ci.yml`) runs this same configure/build/test sequence on every push and PR to `main`.

## Run

Titan needs a Llama-family GGUF model (e.g. TinyLlama, F16/F32 — quantized types are not yet supported):

```sh
./build/bin/titan_generate -m path/to/tinyllama.gguf -p "Once upon a time" -n 64
#   -g            greedy decoding
#   -t 0.8        temperature   --top-k 40   --top-p 0.95   --repeat-penalty 1.1
#   -s 42         RNG seed
```

## Repository Structure

```
docs/          architecture, design, benchmarks, profiling, papers, learning notes
runtime/
  tensor/      Tensor, ThreadPool — core math + parallel matmul
  tokenizer/   SentencePiece-style BPE tokenizer
  models/      GGUF.{h,cpp} — mmap-based model loader/parser
  transformer/ Model.{h,cpp} — RoPE, GQA attention, SwiGLU, RMSNorm
  sampling/    Sampler — greedy, temperature, top-k, top-p, repeat penalty
tests/         GoogleTest unit tests (51 tests across 5 suites)
examples/      generate.cpp (CLI), benchmark.cpp
```

## Testing

51 GoogleTest unit tests across tensor ops, the GGUF loader, the tokenizer, transformer forward-pass components, and sampling — run via `ctest` and enforced in CI on every push and pull request to `main`.

## What This Demonstrates

Titan was built to close a specific gap: understanding *how* an LLM actually runs, not just how to call one through an API. That meant writing the mmap-based model loader, the tensor math, and the transformer forward pass (RoPE, GQA, SwiGLU) by hand in C++ rather than wrapping an existing runtime — and documenting each subsystem's design rationale and tradeoffs as it was built. V2 extends this into systems territory: a KV cache, a request scheduler, continuous batching, and an OpenAI-compatible API — the layer that turns "runs a forward pass" into "behaves like a real inference server."

## License

MIT — see [LICENSE](LICENSE).
