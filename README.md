# Titan

A production-style LLM inference runtime built from scratch in modern C++ — no prior C++ experience going in. The goal: understand every layer between an HTTP request and generated tokens, from tensor math up through transformer execution to (eventually) a scheduler and OpenAI-compatible API.

- **[PRD.md](PRD.md)** — V1 scope (CPU inference), target end of August 2026
- **[PRD-v2.md](PRD-v2.md)** — V2 scope (scheduler, continuous batching, OpenAI-compatible API, dashboard), begins after V1

## Status

**Milestone 1 (CPU Runtime)** — complete. Titan loads a Llama-family GGUF model and generates text end-to-end on CPU: tensor engine, SentencePiece BPE tokenizer, GGUF loader (mmap), transformer runtime (RoPE, GQA attention, SwiGLU), and a sampling engine, wired together by a generation CLI. 48 unit tests pass.

**Milestone 2 (Optimization)** — next: benchmark harness, profiling, cache-blocked matmul, thread pool, ARM NEON SIMD, memory pooling. See issues #16–#21.

**V2 (serving layer)** — KV cache, scheduler, continuous batching, OpenAI-compatible API, dashboard. See [PRD-v2.md](PRD-v2.md) and issues #22–#23.

## Build

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Run

Titan needs a Llama-family GGUF model (e.g. TinyLlama, F16/F32 — quantized types are not yet supported). Point the CLI at it:

```sh
./build/bin/titan_generate -m path/to/tinyllama.gguf -p "Once upon a time" -n 64
#   -g            greedy decoding
#   -t 0.8        temperature   --top-k 40   --top-p 0.95   --repeat-penalty 1.1
#   -s 42         RNG seed
```

## Repository Structure

```
docs/          architecture, design, benchmarks, profiling, papers, learning notes
runtime/       tensor, tokenizer, models (GGUF loader), transformer, sampling
tests/         unit tests (GoogleTest)
examples/      generate.cpp — the generation CLI
```

## License

MIT — see [LICENSE](LICENSE).
