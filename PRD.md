# Titan

> **A Production-Style LLM Runtime Built From Scratch**

**Author:** Shrestha Mishra
**Status:** Planning
**Version:** 1.0
**Duration:** ~6–8 Weeks
**Language:** C++20, CUDA, Python (Tooling), TypeScript (Dashboard)

---

# Vision

Titan is a production-inspired Large Language Model runtime built completely from scratch with the goal of understanding **every major subsystem that powers modern LLM inference**.

Unlike most AI projects, Titan is **not** another chatbot, RAG application, AI agent, or wrapper around an API.

Instead, Titan focuses on the engineering challenges that exist **underneath** ChatGPT.

The project begins with a tensor library and gradually evolves into a complete inference runtime capable of loading Llama models, executing transformer inference, managing GPU memory, scheduling concurrent requests, exposing an OpenAI-compatible API, and visualizing runtime performance.

The ultimate objective is educational.

By the end of this project, I should understand—not just use—the engineering techniques employed by modern AI companies including OpenAI, Anthropic, xAI, NVIDIA, Together AI, Fireworks AI, and vLLM.

Titan should serve as both a learning platform and a showcase of AI systems engineering.

---

# Mission Statement

Build every important component of a modern LLM runtime from first principles while documenting every architectural decision, optimization, benchmark, and tradeoff.

The repository should teach someone how LLM inference actually works under the hood.

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

Titan exists to bridge that gap.

---

# Project Goals

## Primary Goal

Develop a deep understanding of modern LLM systems by implementing them from scratch.

---

## Educational Goals

Understand

- Transformer execution
- Tensor computation
- GPU programming
- CUDA
- Runtime systems
- Scheduling
- KV cache management
- Quantization
- FlashAttention
- Memory management
- Model loading
- Tokenization
- Sampling
- Performance profiling
- Production inference systems

---

## Engineering Goals

Build software that resembles a real production system.

Every subsystem should have

- clear interfaces
- modular architecture
- documentation
- benchmarks
- tests

The project should feel like an open-source runtime rather than a collection of scripts.

---

## Performance Goals

Every optimization should be measurable.

Never optimize blindly.

Profile first.

Measure improvements.

Document results.

---

# Non Goals

Titan is intentionally NOT

- a chatbot
- a RAG framework
- an AI agent
- an MCP server
- a fine tuning library
- a distributed training framework
- a LangChain replacement
- a vector database
- an orchestration framework

These are solved problems and do not align with the educational goals of this project.

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

# High-Level Architecture

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

 Tensor Engine + CUDA Kernels

                   │

      CPU Backend / GPU Backend

                   │

               Hardware
```

---

# Core Components

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

Future

- SIMD
- Mixed precision
- GPU tensors

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

The heart of Titan.

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

GPU acceleration.

Implement custom kernels for

- MatMul
- LayerNorm
- Softmax
- Attention
- memory copies

Future

- FlashAttention
- Tensor Core kernels
- kernel fusion

---

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

## Sampling Engine

Implement

- greedy decoding
- temperature sampling
- top-k
- top-p
- repetition penalty

Future

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
- GPU utilization
- memory usage
- KV cache usage
- scheduler timeline
- request lifecycle

---

## Benchmark Suite

Benchmark against

- PyTorch
- llama.cpp
- Titan CPU
- Titan CUDA

Measure

- tokens/sec
- latency
- throughput
- memory
- GPU utilization

---

# Development Roadmap

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

---

## Milestone 3

CUDA Runtime

Deliverables

- GPU tensors
- CUDA kernels
- memory manager

Goal

Run transformer inference on GPU.

---

## Milestone 4

Inference Runtime

Deliverables

- KV cache
- scheduler
- streaming
- batching

Goal

Serve multiple concurrent requests.

---

## Milestone 5

Production Runtime

Deliverables

- dashboard
- metrics
- OpenAI API
- benchmark reports

Goal

Allow OpenAI SDKs to communicate directly with Titan.

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

runtime/
tensor/
cuda/
memory/
scheduler/
sampling/
models/
tokenizer/
api/

dashboard/

tests/

examples/
```

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

The repository should be educational enough that another engineer could learn modern inference simply by reading the documentation.

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

Issues should be grouped into Milestones corresponding to the development roadmap.

Every merged issue should leave the repository in a working state.

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
- FlashAttention
- FlashAttention-2
- vLLM
- LLM.int8()
- GPTQ
- AWQ
- CUDA Programming Guide
- llama.cpp
- GGUF Specification

Every implementation should reference the paper or source that inspired it.

---

# Success Criteria

Titan is successful if

- It can load and execute a pretrained Llama-family model.
- It exposes an OpenAI-compatible API.
- It supports CPU and CUDA execution.
- It demonstrates measurable performance improvements through optimization.
- Every subsystem is thoroughly documented.
- The repository resembles a professional open-source systems project.
- I can confidently explain every stage of the inference pipeline, from HTTP request to GPU execution and token generation.

Most importantly, Titan should transform me from someone who builds AI applications into someone who deeply understands the systems that power modern large language models.

---

# Next Steps

The planning phase should be completed before writing any production code.

Treat this repository like a real open-source systems project from day one.

The implementation process should begin by generating the engineering artifacts that would normally exist before development starts.

Follow the structured planning workflow:

1. Architecture Documentation
2. GitHub Issues
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

The goal is not simply to build a working LLM runtime, but to develop a deep understanding of every subsystem that powers modern large language model inference.
