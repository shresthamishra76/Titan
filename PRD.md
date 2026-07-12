# Titan (V1)

> **A Production-Style LLM Runtime Built From Scratch — Phase 1: CPU + Basic CUDA Inference**

**Author:** Shrestha Mishra
**Status:** Planning
**Version:** 1.0
**Duration:** ~6–8 Weeks
**Target:** End of August 2026
**Language:** C++20, CUDA, Python (Tooling)

> This is **V1** of a two-phase project. It covers Milestones 1–3 (CPU runtime, optimization, basic CUDA). The full production-serving vision (scheduler, continuous batching, OpenAI-compatible API, dashboard) is scoped separately in [PRD-v2.md](PRD-v2.md) and begins after V1 ships.

---

# Vision

Titan is a production-inspired Large Language Model runtime built completely from scratch with the goal of understanding **every major subsystem that powers modern LLM inference**.

Unlike most AI projects, Titan is **not** another chatbot, RAG application, AI agent, or wrapper around an API.

Instead, Titan focuses on the engineering challenges that exist **underneath** ChatGPT.

V1 takes the project from a tensor library to a working transformer inference engine — first on CPU, then accelerated on GPU with hand-written CUDA kernels. It stops short of production serving (scheduler, batching, API layer) — that's V2.

The ultimate objective is educational, and personal.

By the end of V1, I should understand — not just use — the tensor math, memory layout, and GPU programming techniques employed by modern AI companies including OpenAI, Anthropic, xAI, NVIDIA, Together AI, Fireworks AI, and vLLM.

---

# Mission Statement

Build the CPU and basic CUDA inference core of a modern LLM runtime from first principles while documenting every architectural decision, optimization, benchmark, and tradeoff.

The repository should teach someone how LLM inference actually works under the hood, and double as visible evidence of the author going from zero C++ experience to writing GPU kernels.

---

# Why This Project Exists

Modern software engineers can build applications using LLM APIs.

Machine learning engineers can train transformers.

Few engineers truly understand everything between

```
HTTP Request
      ↓
Scheduler
      ↓
Runtime
      ↓
GPU Kernels
      ↓
Transformer Execution
      ↓
Generated Tokens
```

V1 of Titan builds the bottom half of that stack: Runtime → GPU Kernels → Transformer Execution → Generated Tokens. The top half (HTTP Request → Scheduler) is V2.

---

# Learning Track

I have **no prior C++ experience** going into this project. C++ fluency is a first-class goal of V1, not an assumed prerequisite — Milestone 1 issues are sequenced to build language fundamentals before they're needed for harder work:

1. **Milestone 1 (tensor library, tokenizer, loader)** — basic syntax, classes, RAII, ownership, `std::vector`/`std::unique_ptr`, references vs pointers, build systems (CMake).
2. **Milestone 2 (optimization)** — templates, move semantics, threading (`std::thread`, mutexes), SIMD intrinsics.
3. **Milestone 3 (CUDA)** — CUDA C++ specifics (device/host memory, kernel launch syntax, grid/block indexing) — introduced only after core C++ fundamentals from 1–2 are solid.

Reference material alongside the papers below:
- [cppreference.com](https://en.cppreference.com/) — canonical language/standard-library reference
- *Effective Modern C++* (Scott Meyers) — idiomatic modern C++ (move semantics, smart pointers, templates)
- NVIDIA CUDA C++ Programming Guide (already listed under Papers & References)

Each Milestone 1–2 issue should note which C++ concept it's teaching, so the commit history reads as a legible progression, not just a feature list.

---

# Portfolio / Recruiting Framing

V1 is explicitly calibrated to produce evidence of **MTS (Member of Technical Staff)-level systems ability** for companies like OpenAI, xAI, and Anthropic — not just "an LLM ran on my laptop."

Concretely, that means:

- Every issue and PR should demonstrate a specific competency a systems interviewer would care about: memory management, numerical correctness, performance measurement, low-level hardware understanding (CPU cache behavior, GPU memory hierarchy).
- Optimization work (Milestone 2–3) must show **before/after benchmarks**, not just "made it faster." Reviewers should be able to see the profiling data that motivated each change.
- The repo is **open-sourced from day one**. Commit messages, PR descriptions, and issue writeups are public artifacts — write them for an external technical reader, not as private notes to self. See GitHub Workflow below.
- The C++-novice-to-CUDA-kernels arc (see Learning Track) is itself part of the narrative — don't hide the learning curve, document it. A reviewer seeing deliberate, well-explained progression is more compelling than a repo that looks like it appeared fully-formed.

---

# Project Goals

## Primary Goal

Develop a deep understanding of modern LLM inference systems — and of C++ itself — by implementing a CPU and CUDA inference engine from scratch.

---

## Educational Goals

Understand

- C++ (from first principles)
- Transformer execution
- Tensor computation
- CPU performance (SIMD, threading, cache behavior)
- GPU programming
- CUDA
- Model loading
- Tokenization
- Sampling
- Performance profiling

---

## Engineering Goals

Build software that resembles a real production system, even at V1 scope.

Every subsystem should have

- clear interfaces
- modular architecture
- documentation
- benchmarks
- tests

The project should feel like an open-source runtime rather than a collection of scripts — because it will be read as one.

---

## Performance Goals

Every optimization should be measurable.

Never optimize blindly.

Profile first.

Measure improvements.

Document results.

---

# Non Goals (V1)

V1 of Titan is intentionally NOT

- a chatbot
- a RAG framework
- an AI agent
- an MCP server
- a fine tuning library
- a distributed training framework
- a LangChain replacement
- a vector database
- an orchestration framework
- a production request-serving system (see below — deferred to V2)

---

# Out of Scope for V1 (see PRD-v2.md)

The following are part of the full Titan vision but are explicitly deferred to V2, to keep V1 achievable by end of August 2026:

- KV Cache
- Scheduler / continuous batching / request lifecycle management
- Advanced sampling (speculative decoding)
- OpenAI-compatible API layer
- Dashboard
- Benchmark suite against external frameworks (PyTorch, llama.cpp) — V1 benchmarks Titan CPU vs Titan CUDA only

---

# Guiding Principles

## Learn by Building

If a subsystem is important to understanding modern inference, implement it.

---

## Understand Before Optimizing

Every optimization must have a reason.

Every benchmark must have an explanation.

---

## Measure Everything

Every feature should be benchmarked.

Performance improvements should be reproducible.

---

## Build Like Production

This is not a school project.

Every component should resemble how production software is structured.

---

## Document Everything

Every subsystem should include

- architecture
- rationale
- implementation notes
- future improvements
- references

---

# High-Level Architecture (V1)

```
        Generation Loop (CLI / examples)

                   │

        Transformer Execution

                   │

    KV-free Attention + Sampling

                   │

 Tensor Engine + CUDA Kernels

                   │

      CPU Backend / GPU Backend

                   │

               Hardware
```

The Scheduler, API Layer, and Dashboard shown in the full architecture (PRD-v2.md) sit above this and are not built in V1.

---

# Core Components (V1)

## Tensor Engine

The mathematical foundation of Titan.

Responsibilities

- Tensor abstraction
- Memory layout
- Matrix multiplication
- Broadcasting
- Reshape
- Transpose
- Softmax
- LayerNorm
- GELU

Future (V2+)

- Mixed precision
- Advanced GPU tensor features (see CUDA Backend)

---

## Model Loader

Load pretrained language models.

Support

- GGUF

Future

- SafeTensors

Responsibilities

- weight loading
- tokenizer loading
- metadata parsing
- model configuration

---

## Tokenizer

Implement tokenization from scratch.

Features

- Byte Pair Encoding
- vocabulary lookup
- special tokens
- streaming decode

Future

- SentencePiece

---

## Transformer Runtime

The heart of Titan V1.

Responsible for

- embeddings
- rotary position embeddings
- attention
- feed forward networks
- residual connections
- normalization
- generation loop

---

## CUDA Backend

Basic GPU acceleration — the first hands-on CUDA work in the project.

Implement custom kernels for

- MatMul
- LayerNorm
- Softmax
- Attention
- memory copies

Future (V2+)

- FlashAttention
- Tensor Core kernels
- kernel fusion

---

## Sampling Engine (basic)

Implement

- greedy decoding
- temperature sampling
- top-k
- top-p
- repetition penalty

Future (V2)

- speculative decoding

---

## Benchmark Suite (V1 scope)

Benchmark

- Titan CPU
- Titan CUDA

Measure

- tokens/sec
- latency
- memory
- GPU utilization

External comparisons (PyTorch, llama.cpp) are V2 scope.

---

# Development Roadmap (V1)

## Milestone 1

CPU Runtime

Deliverables

- tensor library
- tokenizer
- GGUF loader
- transformer execution
- text generation

Goal

Generate text from TinyLlama entirely on CPU.

C++ concepts introduced: classes, RAII, ownership (`unique_ptr`/`shared_ptr`), references vs pointers, CMake build system.

---

## Milestone 2

Optimization

Deliverables

- thread pool
- SIMD
- profiling
- benchmarks

Goal

Optimize CPU inference through measurement.

C++ concepts introduced: templates, move semantics, `std::thread`/synchronization, SIMD intrinsics.

---

## Milestone 3

CUDA Runtime

Deliverables

- GPU tensors
- CUDA kernels
- memory manager

Goal

Run transformer inference on GPU.

C++/CUDA concepts introduced: device/host memory management, kernel launch configuration, grid/block/thread indexing.

---

**Milestones 4 and 5 (Inference Runtime, Production Runtime) are V2 scope — see [PRD-v2.md](PRD-v2.md).**

---

# Repository Structure

```
titan/

docs/
architecture/
design/
benchmarks/
profiling/
papers/
learning/          # C++/CUDA notes as I learn each concept

runtime/
tensor/
cuda/
memory/
sampling/
models/
tokenizer/

tests/

examples/
```

Note: `scheduler/`, `api/`, and `dashboard/` are added in V2, not created in V1.

---

# Documentation Requirements

Every major subsystem should include

- Overview
- Design rationale
- Architecture diagrams
- Public interfaces
- Benchmarks
- Tradeoffs
- References
- Future improvements

The repository should be educational enough that another engineer could learn modern inference simply by reading the documentation — and a recruiter or technical interviewer should be able to read the commit/PR history and understand the depth of what was built.

---

# GitHub Workflow

Titan should be developed like a real production open-source project.

Development should **never** happen directly from a to-do list.

Instead, every piece of work should be represented by a GitHub Issue.

Each issue should include

- Problem statement
- Motivation
- Acceptance criteria
- Implementation notes
- References
- Related papers
- Stretch goals
- (Milestone 1–2 only) Which C++ concept this issue teaches

Issues should be grouped into Milestones corresponding to the development roadmap (1, 2, 3 for V1).

Every merged issue should leave the repository in a working state.

**Because this repo is open-sourced,** commit messages, PR descriptions, and issue writeups are public-facing artifacts. Write them for an external technical reader — clear problem statement, what was tried, what was learned, what the benchmark showed — not as private shorthand notes.

---

# Engineering Standards

Every feature must include

- implementation
- tests
- benchmarks
- documentation
- profiling (where applicable)

No feature is considered complete without all five.

---

# Papers & References

Each subsystem should be implemented only after studying the relevant literature.

Core references include

- Attention Is All You Need
- RoFormer
- FlashAttention (read for context even though V1 doesn't implement it)
- CUDA Programming Guide
- llama.cpp
- GGUF Specification

Every implementation should reference the paper or source that inspired it.

---

# Success Criteria (V1)

Titan V1 is successful if

- It can load and execute a pretrained Llama-family model (TinyLlama) entirely on CPU.
- It can run the same inference path on CUDA with hand-written kernels.
- It demonstrates measurable, documented performance improvement from CPU baseline → optimized CPU → CUDA.
- Every V1 subsystem (tensor engine, model loader, tokenizer, transformer runtime, CUDA backend) is thoroughly documented.
- The repository resembles a professional open-source systems project, with commit/issue history that reads as a credible C++ learning + systems engineering narrative.
- I can confidently explain every stage of the V1 pipeline, from tensor math to GPU kernel execution.
- The result is something I can point to in an MTS interview at OpenAI/xAI/Anthropic-tier companies as evidence of systems ability.

---

# Next Steps

The planning phase should be completed before writing any production code.

Treat this repository like a real open-source systems project from day one.

The implementation process should begin by generating the engineering artifacts that would normally exist before development starts.

Follow the structured planning workflow:

1. Architecture Documentation
2. GitHub Issues (Milestones 1–3)
3. GitHub Milestones
4. GitHub Labels
5. Dependency Graph
6. Kanban Board

---

# Development Philosophy

Development should always follow this workflow:

1. Read the relevant research paper(s).
2. Understand the underlying concepts.
3. Review the architecture documentation.
4. Implement the GitHub Issue.
5. Write tests.
6. Benchmark performance.
7. Document findings.
8. Open a pull request.
9. Review and merge.

The goal is not simply to build a working CPU/CUDA inference engine, but to develop a deep understanding of every subsystem involved — and to come out the other side fluent in C++ and ready to demonstrate that fluency to top-tier AI infrastructure teams.
