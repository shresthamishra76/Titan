# Subsystem Dependency Graph

This graph defines the **build order** for Titan. An arrow `A --> B` means *B depends on A* (A must exist and be correct before B can be finished). It is the source of truth for how the GitHub issues are sequenced across milestones.

```mermaid
graph TD
    subgraph M1["Milestone 1 — CPU Runtime"]
        TENSOR["Tensor Engine<br/>runtime/tensor"]
        TOK["Tokenizer (BPE)<br/>runtime/tokenizer"]
        LOADER["GGUF Loader<br/>runtime/models"]
        XFMR["Transformer Runtime<br/>runtime/transformer"]
        SAMPLE["Sampling Engine<br/>runtime/sampling"]
        CLI["Generation CLI<br/>examples/generate"]
    end

    subgraph M2["Milestone 2 — Optimization"]
        BENCH["Benchmark Harness"]
        PROF["Profiling"]
        MATMUL["Matmul Opt (cache blocking)"]
        THREADS["Thread Pool<br/>runtime/memory"]
        SIMD["SIMD (ARM NEON)"]
        MEM["Memory Pooling"]
    end

    subgraph V2["V2 — Serving Layer (outlined)"]
        KV["KV Cache<br/>runtime/cache"]
        SCHED["Scheduler<br/>scheduler/"]
        BATCH["Continuous Batching"]
        API["OpenAI API<br/>api/"]
        DASH["Dashboard<br/>dashboard/"]
    end

    TENSOR --> LOADER
    TENSOR --> XFMR
    TOK --> XFMR
    LOADER --> XFMR
    XFMR --> SAMPLE
    XFMR --> CLI
    SAMPLE --> CLI

    CLI --> BENCH
    BENCH --> PROF
    PROF --> MATMUL
    MATMUL --> THREADS
    THREADS --> SIMD
    SIMD --> MEM

    XFMR -.V1 revision.-> KV
    KV --> SCHED
    SCHED --> BATCH
    BATCH --> API
    API --> DASH
```

## Critical path to first token (Milestone 1)

```
Tensor Engine ──▶ GGUF Loader ──▶ Transformer Runtime ──▶ Sampling ──▶ CLI
                                    ▲
Tokenizer ──────────────────────────┘
```

- **Tensor Engine** is the root dependency — everything numerical sits on it, so it lands first.
- **Tokenizer** is independent of the tensor engine and can be built in parallel; its final wiring depends only on the GGUF metadata parser (for the embedded vocab).
- **GGUF Loader** depends on the Tensor engine (to wrap loaded weights as tensor views).
- **Transformer Runtime** is the convergence point: it needs the tensor engine, the loaded weights, and the tokenizer.
- **Sampling** and the **CLI** sit on top of the forward pass.

## Milestone 2 ordering

Optimization is strictly **profile-first**: the benchmark harness and profiler come before any optimization so that every change (matmul blocking → threading → SIMD → memory pooling) is justified by before/after data.

## V2 revisions to V1

Two V1 subsystems get revised (not rewritten) in V2, and these are called out explicitly in their issues:

1. **Attention** (`runtime/transformer`) gains a KV-cache read/append path.
2. **Forward pass** gains batched/ragged execution to support continuous batching.
