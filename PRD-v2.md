# Titan (V2)

> **A Production-Style LLM Runtime Built From Scratch — Phase 2: Full Inference Serving**

**Author:** Shrestha Mishra
**Status:** Planning (begins after V1 complete)
**Version:** 2.0
**Duration:** Open-ended
**Language:** C++20, Python (Tooling), TypeScript (Dashboard)

> This is **V2**, the second phase of Titan. It assumes [PRD.md (V1)](PRD.md) is complete: a working tensor engine, GGUF model loader, tokenizer, and CPU transformer inference. V2 turns that inference core into a real production-style serving system — request scheduling, continuous batching, an OpenAI-compatible API, and a live dashboard.

---

# Vision

Where V1 proved I could execute a transformer forward pass on CPU, V2 turns that into something that behaves like an actual inference server: multiple concurrent requests, a KV cache, continuous batching, streaming responses, and an OpenAI-compatible API surface that real client SDKs can talk to.

This is the layer that sits between "I can run a model" and "I built something that resembles vLLM, TGI, or an internal serving stack at a frontier lab."

```
HTTP Request
      ↓
Scheduler          ← V2
      ↓
Runtime            ← V1 (done)
      ↓
Transformer Execution  ← V1 (done)
      ↓
Generated Tokens
```

---

# Mission Statement

Build the request-serving layer of a modern LLM runtime — KV cache, scheduler, continuous batching, OpenAI-compatible API, and observability dashboard — on top of the V1 inference core, documenting every architectural decision, optimization, benchmark, and tradeoff.

---

# Relationship to V1

V2 does not repeat V1 scope. It assumes as already built and stable:

- Tensor Engine (CPU)
- Model Loader (GGUF)
- Tokenizer (BPE)
- Transformer Runtime (single-request forward pass, CPU)
- Basic Sampling Engine (greedy, temperature, top-k, top-p)

V2 adds the layers above and around that core. If V1 subsystems need rework to support batching/KV cache (e.g., the transformer runtime's forward-pass interface), that rework is in-scope for V2 but should be called out explicitly in the relevant issue as a "V1 revision," not silently redone.

---

# Project Goals

## Primary Goal

Develop a deep understanding of production LLM serving systems by implementing the scheduling, batching, caching, and API layers from scratch.

---

## Educational Goals

Understand

- Runtime systems
- Scheduling
- KV cache management
- Quantization
- Continuous batching
- Production inference systems
- Observability for ML systems

---

## Engineering Goals

Same standard as V1: every subsystem should have clear interfaces, modular architecture, documentation, benchmarks, and tests. By V2 the project should be indistinguishable in structure from a real open-source inference server.

---

## Performance Goals

Same discipline as V1 — profile first, measure improvements, document results. V2 adds comparative benchmarking against established frameworks (PyTorch, llama.cpp), which V1 explicitly deferred.

---

# Non Goals

Titan (V2, same as V1) is intentionally NOT

- a chatbot
- a RAG framework
- an AI agent
- an MCP server
- a fine tuning library
- a distributed training framework
- a LangChain replacement
- a vector database
- a general orchestration framework

---

# Guiding Principles

Same as V1: Learn by Building, Understand Before Optimizing, Measure Everything, Build Like Production, Document Everything. See PRD.md for full descriptions — unchanged for V2.

---

# High-Level Architecture (Full System)

```
                Client

                   │

      OpenAI Compatible API

                   │

          Request Scheduler

                   │

      Continuous Batching

                   │

        Generation Runtime

                   │

        Sampling Engine

                   │

     Transformer Execution

                   │

    KV Cache + Memory Manager

                   │

        Tensor Engine

                   │

          CPU Backend

                   │

               Hardware
```

Everything below "KV Cache + Memory Manager" is V1. Everything from "KV Cache + Memory Manager" up is V2.

---

# Core Components (V2)

## KV Cache

Store attention history.

Responsibilities

- append
- lookup
- reuse
- eviction
- memory pooling

Future

- paged KV cache

---

## Scheduler

Manage inference requests.

Features

- request lifecycle
- continuous batching
- prioritization
- streaming
- cancellation

Future

- adaptive batching
- fairness algorithms

---

## Sampling Engine (advanced)

Builds on V1's basic sampling engine.

Add

- speculative decoding

---

## API Layer

Expose

- OpenAI compatible endpoints
- streaming
- health checks
- metrics

---

## Dashboard

Visualize

- latency
- throughput
- memory usage
- KV cache usage
- scheduler timeline
- request lifecycle

---

## Benchmark Suite (full)

Benchmark against

- PyTorch
- llama.cpp
- Titan CPU

Measure

- tokens/sec
- latency
- throughput
- memory

---

# Development Roadmap (V2)

## Milestone 3

Inference Runtime

Deliverables

- KV cache
- scheduler
- streaming
- batching

Goal

Serve multiple concurrent requests.

---

## Milestone 4

Production Runtime

Deliverables

- dashboard
- metrics
- OpenAI API
- benchmark reports

Goal

Allow OpenAI SDKs to communicate directly with Titan.

---

# Repository Structure (additions for V2)

```
titan/

runtime/
scheduler/       # new in V2
api/              # new in V2

dashboard/        # new in V2
```

All other directories (tensor/, memory/, models/, tokenizer/, tests/, examples/, docs/) already exist from V1 and are extended, not recreated.

---

# Documentation Requirements

Same as V1 — every major subsystem needs Overview, Design rationale, Architecture diagrams, Public interfaces, Benchmarks, Tradeoffs, References, Future improvements.

---

# GitHub Workflow

Same process as V1: every piece of work is a GitHub Issue with problem statement, motivation, acceptance criteria, implementation notes, references, related papers, stretch goals. Issues grouped into Milestones 3 and 4. Every merged issue leaves the repo in a working state.

Since the repo is open-sourced and V1's commit/issue history already reads as a portfolio narrative, keep that standard for V2: PRs and issues should read as evidence of production-systems thinking (scheduling, concurrency, observability) to an external technical reviewer.

---

# Engineering Standards

Same as V1: every feature needs implementation, tests, benchmarks, documentation, and profiling (where applicable). No feature is complete without all five.

---

# Papers & References

Builds on V1's reading list. Additional core references for V2:

- vLLM
- LLM.int8()
- GPTQ
- AWQ

(Attention Is All You Need, RoFormer, FlashAttention, llama.cpp, GGUF Spec carried over from V1.)

---

# Success Criteria (V2)

Titan V2 is successful if

- It exposes an OpenAI-compatible API that real client SDKs can talk to.
- It serves multiple concurrent requests via a working scheduler and continuous batching.
- KV cache management is implemented and measurably improves throughput.
- Comparative benchmarks against PyTorch and llama.cpp are documented.
- The dashboard gives real-time visibility into scheduler, KV cache, and runtime state.
- Every V2 subsystem is thoroughly documented, to the same standard as V1.
- I can confidently explain the entire pipeline, from HTTP request through scheduling, batching, KV cache, and CPU execution, to token generation.

Most importantly, Titan V2 completes the transformation from someone who builds AI applications into someone who deeply understands the systems that power modern large language model inference — end to end, request to token.

---

# Development Philosophy

Same workflow as V1: read the paper, understand the concept, review architecture docs, implement the issue, write tests, benchmark, document findings, open a PR, review and merge.
