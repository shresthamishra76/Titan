# Titan

A production-style LLM inference runtime built from scratch in C++/CUDA — no prior C++ experience going in. The goal: understand every layer between an HTTP request and generated tokens, from tensor math up through GPU kernels to (eventually) a scheduler and OpenAI-compatible API.

- **[PRD.md](PRD.md)** — V1 scope (CPU + basic CUDA inference), target end of August 2026
- **[PRD-v2.md](PRD-v2.md)** — V2 scope (scheduler, continuous batching, OpenAI-compatible API, dashboard), begins after V1

## Status

**Milestone 1 (CPU Runtime)** — not started. Repo scaffold in place.

## Build

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

CUDA backend is off by default. Enable with `-DTITAN_ENABLE_CUDA=ON` once the CUDA toolkit and Milestone 3 work land.

## Repository Structure

```
docs/          architecture, design, benchmarks, profiling, papers, learning notes
runtime/       tensor, cuda, memory, sampling, models, tokenizer
tests/         unit tests (GoogleTest)
examples/      example programs
```

## License

MIT — see [LICENSE](LICENSE).
