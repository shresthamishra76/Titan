# Architecture Documentation

This directory documents Titan's architecture, one file per subsystem. Each subsystem doc covers: overview, design rationale, diagrams, public interfaces, benchmarks, tradeoffs, references, and future improvements. Docs land alongside the implementation of each subsystem — see [PRD.md](../../PRD.md).

## Index

| Doc | Subsystem | Status |
| --- | --- | --- |
| [dependency-graph.md](dependency-graph.md) | Build order across all milestones | ✅ |
| [tensor.md](tensor.md) | Tensor Engine (Milestone 1) | ✅ |
| tokenizer.md | Tokenizer / BPE (Milestone 1) | ⏳ |
| model-loader.md | GGUF Loader (Milestone 1) | ⏳ |
| transformer-runtime.md | Transformer Runtime (Milestone 1) | ⏳ |
| sampling.md | Sampling Engine (Milestone 1) | ⏳ |

## High-level architecture (V1)

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

The tokenizer feeds token IDs into the runtime; the GGUF loader supplies the weights. The scheduler, API layer, and dashboard from the full vision sit above this and are V2 scope — see [PRD-v2.md](../../PRD-v2.md).

## Target model

TinyLlama (Llama-family, GGUF). All hyperparameters — layer count, head counts, hidden/FFN dims, RoPE frequency base, RMSNorm epsilon, context length, vocab — are **read from GGUF metadata at load time**, never hard-coded. Because the target is a Llama-architecture model, the runtime implements the Llama operator set: **RMSNorm** (not LayerNorm), **SiLU/SwiGLU** (not GELU), **grouped-query attention** (GQA), and **rotary position embeddings** (RoPE).
